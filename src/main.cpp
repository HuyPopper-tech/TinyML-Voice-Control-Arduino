/* * COMBINED FIRMWARE
 * 1. Engine: Edge Impulse SDK Continuous Inference (High Performance DSP)
 * 2. Logic: Finite State Machine & Voting Mechanism (High Stability)
 */

#include <Arduino.h>
#include <PDM.h>
#include <Speech_Recognition_inferencing.h>
#include "mbed.h"

/* ========== CLASSIFIER CONFIGURATION (From SDK) ==========
   - EIDSP_QUANTIZE_FILTERBANK: RAM optimization flag
   - Divides sample window (typically 1000ms) into 5 slices
   - Each processing handles 200ms of new data (INFERENCE_EVERY_MS = 200)
*/
#define EIDSP_QUANTIZE_FILTERBANK                 0
#define EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW     5

/* ========== APPLICATION CONFIGURATION  ========== */
#define MAX_LABEL_LEN                                     6
#define LISTENING_TIMEOUT_MS                              10000
#define DEVICE_TIMEOUT_MS                                 4000
#define PREDICTION_HISTORY_SIZE                           5
#define VOTING_THRESHOLD                                  3

/* ========== PIN DEFINITIONS ========== */
#define FAN_IN1               D3
#define FAN_IN2               D4

/* ========== AUDIO BUFFER STRUCTURE (From SDK Continuous) ==========
   Double-buffering structure for continuous inference with PDM input
*/
typedef struct {
    signed short *buffers[2];
    volatile unsigned char buf_select;
    volatile unsigned char buf_ready;
    volatile unsigned int buf_count;
    unsigned int n_samples;
} inference_t;

static inference_t inference;
static bool record_ready = false;
static signed short *sampleBuffer;
static bool debug_nn = false;

/* ========== GLOBAL VARIABLES: FSM & VOTING ==========
   Manages system states, prediction history, and voting mechanism
*/

enum SystemState : uint8_t { 
    STATE_IDLE, 
    STATE_WAIT_ACTION, 
    STATE_DEVICE_ON, 
    STATE_DEVICE_OFF 
};
SystemState current_state = STATE_IDLE;

static unsigned long last_wake_time = 0;
static char last_label_processed[MAX_LABEL_LEN] = {0};
static unsigned long last_label_time = 0;

static int prediction_history[PREDICTION_HISTORY_SIZE];
static int history_index = 0;
static bool history_filled = false;

/* ========== FUNCTION DECLARATIONS ========== */

void process_fsm(const char* label, float confidence);
void RGB_control(bool red, bool green, bool blue);
void fan_control(bool on);
static bool microphone_inference_start(uint32_t n_samples);
static bool microphone_inference_record(void);
static void microphone_inference_end(void);
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr);
static void pdm_data_ready_inference_callback(void);
void print_memory_usage(void);

void setup() {
    /* Initialize serial communication (115200 baud or 921600 optional) */
    Serial.begin(921600);

    Serial.println("Edge Impulse Continuous + FSM System");

    /* ===== STEP 1: Configure I/O Pins ===== */
    pinMode(LED_BUILTIN, OUTPUT);
    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    pinMode(FAN_IN1, OUTPUT);
    pinMode(FAN_IN2, OUTPUT);

    /* Set initial state: Idle (Red LED) */
    RGB_control(true, false, false);

    /* ===== STEP 2: Initialize & Display Model Settings ===== */
    ei_printf("Inferencing settings:\n");
    ei_printf("\tInterval: %.2f ms.\n", (float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    ei_printf("\tSlices per window: %d\n", EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);

    run_classifier_init();

    /* ===== STEP 3: Initialize Microphone (Slice-based Continuous) =====
       Slice size = Total samples / 5 (e.g., 16000 / 5 = 3200 samples = 200ms)
    */
    if (microphone_inference_start(EI_CLASSIFIER_SLICE_SIZE) == false) {
        ei_printf("ERR: Could not allocate audio buffer!\r\n");
        return;
    }
    print_memory_usage();
}

void loop() {
    /* ===== SECTION 1: CONTINUOUS AUDIO PROCESSING =====
       Records microphone data in slices (blocks until 200ms collected)
       Each iteration processes one new slice of audio data
    */
    bool m = microphone_inference_record();
    if (!m) {
        ei_printf("ERR: Failed to record audio...\n");
        return;
    }

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
    /* get_data callback: SDK uses this to concatenate new data with previous data */
    signal.get_data = &microphone_audio_signal_get_data; 
    ei_impulse_result_t result = {0};

    /* Run classifier in CONTINUOUS (Stateful) mode
       Automatically handles sliding window overlap across slices
    */
    EI_IMPULSE_ERROR r = run_classifier_continuous(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", r);
        return;
    }

    /* ===== SECTION 2: TIMEOUT HANDLING (Checked every loop) ===== */
    unsigned long current_time = millis();
    
    /* Return to IDLE if listening timeout exceeded */
    if (current_state != STATE_IDLE && (current_time - last_wake_time > LISTENING_TIMEOUT_MS)) {
        current_state = STATE_IDLE;
        RGB_control(true, false, false);
        ei_printf("--- Timeout: Return to IDLE ---\n");
    }

    /* Return to wait for action if device timeout exceeded */
    if ((current_state == STATE_DEVICE_ON || current_state == STATE_DEVICE_OFF) &&
        (current_time - last_wake_time > DEVICE_TIMEOUT_MS)) {
        current_state = STATE_WAIT_ACTION;
        RGB_control(false, false, true);
        ei_printf("--- Device Timeout: Return to WAIT_ACTION ---\n");
    }

    /* ===== SECTION 3: RESULT PROCESSING & VOTING MECHANISM =====
       Find the label with highest confidence score in this inference cycle
    */
    int best_idx = -1;
    float best_val = 0;
    
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        if (result.classification[ix].value > best_val) {
            best_val = result.classification[ix].value;
            best_idx = ix;
        }
    }

    /* Push current prediction into circular history buffer */
    prediction_history[history_index] = best_idx;
    history_index++;
    if (history_index >= PREDICTION_HISTORY_SIZE) {
        history_index = 0;
        history_filled = true;
    }

    /* Voting mechanism: Count occurrences and find consensus */
    if (history_filled) {
        int counts[EI_CLASSIFIER_LABEL_COUNT] = {0};
        for (int i = 0; i < PREDICTION_HISTORY_SIZE; i++) {
            counts[prediction_history[i]]++;
        }

        int winner_idx = -1;
        for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
            if (counts[i] >= VOTING_THRESHOLD) {
                winner_idx = i;
                break; 
            }
        }

        if (winner_idx >= 0) {
            const char* winner_label = result.classification[winner_idx].label;
            float winner_score = result.classification[winner_idx].value;

            /* Light debug output: show inference timing and voting result */
            ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
                result.timing.dsp, result.timing.classification, result.timing.anomaly);
            ei_printf(": \n");
            print_memory_usage();
            ei_printf("Winner: %s (%d/%d)\n", winner_label, counts[winner_idx], PREDICTION_HISTORY_SIZE);

            /* Filter out noise and unknown labels before FSM processing */
            if (strcmp(winner_label, "_noise") != 0 && strcmp(winner_label, "_unknown") != 0) {
                    process_fsm(winner_label, winner_score);
            }
        }
    }
}

/* ========== FINITE STATE MACHINE (FSM) LOGIC ==========
   Processes recognized labels and manages system state transitions
   States: IDLE -> WAIT_ACTION -> (DEVICE_ON or DEVICE_OFF) -> WAIT_ACTION -> IDLE
*/
void process_fsm(const char* label, float confidence) {
    unsigned long current_time = millis();

    /* Debounce: Prevent rapid repeated commands (<500ms interval) */
    if (strncmp(label, last_label_processed, MAX_LABEL_LEN) == 0 && (current_time - last_label_time < 500)) {
        return;
    }

    strncpy(last_label_processed, label, MAX_LABEL_LEN - 1);
    last_label_processed[MAX_LABEL_LEN - 1] = '\0';
    last_label_time = current_time;

    switch (current_state) {
        case STATE_IDLE:
            /* In IDLE state: Only WAKE command transitions to WAIT_ACTION */
            if (strcmp(label, "WAKE") == 0) {
                current_state = STATE_WAIT_ACTION;
                last_wake_time = current_time;
                RGB_control(false, false, true);
                ei_printf(">>> WOKEN UP! Waiting for command...\n");
            }
            break;

        case STATE_WAIT_ACTION:
            /* In WAIT_ACTION state: Accept ON/OFF commands; reset timeout on any command */
            last_wake_time = current_time;
            if (strcmp(label, "ON") == 0) {
                current_state = STATE_DEVICE_ON;
                ei_printf("[FSM] ACTION = ON, awaiting device...\n");
            } else if (strcmp(label, "OFF") == 0) {
                current_state = STATE_DEVICE_OFF;
                ei_printf("[FSM] ACTION = OFF, awaiting device...\n");
            }
            break;

        case STATE_DEVICE_ON:
            /* In DEVICE_ON state: Execute ON commands for devices (LED/FAN) */
            if (strcmp(label, "LED") == 0) {
                ei_printf(">>> EXECUTING: LED ON <<<\n");
                digitalWrite(LED_BUILTIN, HIGH);
                current_state = STATE_WAIT_ACTION;
            } else if (strcmp(label, "FAN") == 0) {
                ei_printf(">>> EXECUTING: FAN ON <<<\n");
                fan_control(true);
                current_state = STATE_WAIT_ACTION;
            }
            break;

        case STATE_DEVICE_OFF:
            /* In DEVICE_OFF state: Execute OFF commands for devices (LED/FAN) */
            if (strcmp(label, "LED") == 0) {
                ei_printf(">>> EXECUTING: LED OFF <<<\n");
                digitalWrite(LED_BUILTIN, LOW);
                current_state = STATE_WAIT_ACTION;
            } else if (strcmp(label, "FAN") == 0) {
                ei_printf(">>> EXECUTING: FAN OFF <<<\n");
                fan_control(false);
                current_state = STATE_WAIT_ACTION;
            }
            break;
    }
}

/* ========== MEMORY USAGE REPORTING FUNCTION ========== */
/**
 * Prints current memory usage statistics (heap and stack)
 * Uses mbed_stats to gather memory information
 * Outputs to serial console
 */
void print_memory_usage() {
    mbed_stats_heap_t heap_stats;
    mbed_stats_heap_get(&heap_stats);

    mbed_stats_stack_t stack_stats;
    mbed_stats_stack_get(&stack_stats);
    ei_printf("Memory Usage:\n");
    ei_printf("Heap - Current: %lu bytes, Max: %lu bytes\n", heap_stats.current_size, heap_stats.max_size);
    ei_printf("Stack - Max Used: %lu bytes\n", stack_stats.max_size);

    ei_printf("Est. RAM Active: %lu bytes\n", heap_stats.current_size + stack_stats.max_size);
    ei_printf("====================\n");
}

/* ========== HARDWARE CONTROL FUNCTIONS ========== */

/* Control RGB LED: Active-low logic (LOW = ON, HIGH = OFF) */
void RGB_control(bool red, bool green, bool blue) {
    digitalWrite(LEDR, red ? LOW : HIGH);
    digitalWrite(LEDG, green ? LOW : HIGH);
    digitalWrite(LEDB, blue ? LOW : HIGH);
}

/* Control DC fan motor via dual-input H-bridge
   ON: FAN_IN1=HIGH, FAN_IN2=LOW (forward rotation)
   OFF: Both LOW (motor disabled)
*/
void fan_control(bool on) {
    if (on) {
        digitalWrite(FAN_IN1, HIGH);
        digitalWrite(FAN_IN2, LOW);
    } else {
        digitalWrite(FAN_IN1, LOW);
        digitalWrite(FAN_IN2, LOW);
    }
}

/* ========== AUDIO PROCESSING FUNCTIONS (From SDK) ==========
   These functions manage double-buffering for continuous inference
   PDM data flows into buffers that are alternately filled and processed
*/

/* PDM Data Ready Callback (ISR)
   Executes when PDM microphone has new audio data available
   - Reads raw PDM samples into temporary buffer
   - Transfers to active inference buffer
   - Toggles buffers when one is filled (double-buffering)
*/
static void pdm_data_ready_inference_callback(void) {
    int bytesAvailable = PDM.available();
    /* Read available PDM data into temporary sample buffer */
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);

    if (record_ready == true) {
        for (int i = 0; i < bytesRead >> 1; i++) {
            /* Push sample into current inference buffer */
            inference.buffers[inference.buf_select][inference.buf_count++] = sampleBuffer[i];

            if (inference.buf_count >= inference.n_samples) {
                /* Buffer full: Toggle to other buffer, signal ready slice */
                inference.buf_select ^= 1;
                inference.buf_count = 0;
                inference.buf_ready = 1;
            }
        }
    }
}

/* Initialize microphone with double-buffering
   Allocates two buffers (n_samples each) for alternating use
   Parameters:
   - n_samples: Size of each inference buffer (typically 4000 = 250ms @ 16kHz)
*/
static bool microphone_inference_start(uint32_t n_samples) {
    /* Allocate first buffer for alternating inference */
    inference.buffers[0] = (signed short *)malloc(n_samples * sizeof(signed short));
    if (inference.buffers[0] == NULL) return false;

    /* Allocate second buffer for alternating inference */
    inference.buffers[1] = (signed short *)malloc(n_samples * sizeof(signed short));
    if (inference.buffers[1] == NULL) {
        free(inference.buffers[0]);
        return false;
    }

    /* Allocate temporary sample buffer (half size due to PDM byte reading) */
    sampleBuffer = (signed short *)malloc((n_samples >> 1) * sizeof(signed short));
    if (sampleBuffer == NULL) {
        free(inference.buffers[0]);
        free(inference.buffers[1]);
        return false;
    }

    /* Initialize inference state variables */
    inference.buf_select = 0;
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    /* Register PDM data ready callback (ISR) */
    PDM.onReceive(&pdm_data_ready_inference_callback);
    
    /* Configure PDM buffer size based on slice size (n_samples) */
    PDM.setBufferSize((n_samples >> 1) * sizeof(int16_t));

    /* Start PDM microphone at configured frequency */
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        ei_printf("Failed to start PDM!");
        return false;
    }
    
    PDM.setGain(127);
    record_ready = true;
    return true;
}

/* Wait for microphone buffer to fill (blocking until next slice ready)
   - Blocks main loop until one slice of audio is collected
   - PDM ISR continues in background, filling buffer
   - Approximately 250ms wait per call
*/
static bool microphone_inference_record(void) {
    bool ret = true;
    if (inference.buf_ready == 1) {
        ei_printf("Error sample buffer overrun.\n");
        ret = false;
    }

    /* Block until PDM ISR signals buffer ready (next slice collected) */
    while (inference.buf_ready == 0) {
        delay(1);
    }

    inference.buf_ready = 0;
    return ret;
}

/* Convert audio data from int16 to float for classifier input
   Uses the non-active buffer (XOR toggles between 0 and 1)
   Parameters:
   - offset: Start position in buffer
   - length: Number of samples to convert
   - out_ptr: Output float array
*/
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
    /* Convert int16 samples to float from inactive buffer for DSP processing */
    numpy::int16_to_float(&inference.buffers[inference.buf_select ^ 1][offset], out_ptr, length);
    return 0;
}

/* Cleanup and deallocate microphone resources */
static void microphone_inference_end(void) {
    PDM.end();
    free(inference.buffers[0]);
    free(inference.buffers[1]);
    free(sampleBuffer);
}
