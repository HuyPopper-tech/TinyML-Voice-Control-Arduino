/* * COMBINED FIRMWARE (Dataset compound words) 
 * 1. Engine: Edge Impulse SDK Continuous Inference
 * 2. Logic: Simple State Machine (WAKE -> ACTIVE)
 */

#include <Arduino.h>
#include <PDM.h>
#include <Speech_Recognition_V2_inferencing.h>

/* ========== CLASSIFIER CONFIGURATION (From SDK) ==========
   - EIDSP_QUANTIZE_FILTERBANK: RAM optimization flag
   - Slices per model window: controls slice size for continuous inferencing
*/
#define EIDSP_QUANTIZE_FILTERBANK               1
#define EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW   4

/* ========== APPLICATION CONFIGURATION ========== */
#define CONFIDENCE_THRESHOLD                    0.75f
#define COMMAND_TIMEOUT_MS                      5000
#define ACTION_DEBOUNCE_MS                      1000

/* ========== PIN DEFINITIONS ========== */
#define FAN_IN1                                 D3
#define FAN_IN2                                 D4

/* ========== AUDIO BUFFER STRUCTURE (From SDK Continuous) ==========
   Double-buffering structure for continuous inference with PDM input
*/
typedef struct {
    signed short *buffers[2];
    unsigned char buf_select;
    unsigned char buf_ready;
    unsigned int buf_count;
    unsigned int n_samples;
} inference_t;

static inference_t inference;
static bool record_ready = false;
static signed short *sampleBuffer;
static bool debug_nn = false;
static int print_results = -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);

/* ========== GLOBAL VARIABLES: STATE MACHINE ========== */
enum SystemState : uint8_t {
    STATE_IDLE,     // Chờ WAKE
    STATE_ACTIVE    // Đã WAKE, chờ lệnh điều khiển
};

static SystemState current_state = STATE_IDLE;
static unsigned long last_activity_time = 0;
static unsigned long last_action_executed_time = 0;

/* ========== FUNCTION DECLARATIONS ========== */
static void set_rgb(bool r, bool g, bool b);
static void fan_control(bool on);
static void execute_command(const char* label);

static bool microphone_inference_start(uint32_t n_samples);
static bool microphone_inference_record(void);
static void microphone_inference_end(void);
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr);
static void pdm_data_ready_inference_callback(void);

void setup() {
    /* Initialize serial communication */
    Serial.begin(921600);
    while (!Serial);
    Serial.println("Edge Impulse Intent Recognition Demo");

    /* ===== STEP 1: Configure I/O Pins ===== */
    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    pinMode(LED_BUILTIN, OUTPUT);
    pinMode(FAN_IN1, OUTPUT);
    pinMode(FAN_IN2, OUTPUT);

    /* Set initial state: IDLE (Red LED) */
    set_rgb(true, false, false);
    fan_control(false);
    digitalWrite(LED_BUILTIN, LOW);

    /* ===== STEP 2: Initialize & Display Model Settings ===== */
    ei_printf("Inferencing settings:\n");
    ei_printf("\tInterval: %.2f ms.\n", (float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    ei_printf("\tNo. of classes: %d\n",
              sizeof(ei_classifier_inferencing_categories) /
              sizeof(ei_classifier_inferencing_categories[0]));

    run_classifier_init();

    /* ===== STEP 3: Initialize Microphone (Slice-based Continuous) ===== */
    if (microphone_inference_start(EI_CLASSIFIER_SLICE_SIZE) == false) {
        ei_printf("ERR: Could not allocate audio buffer (size %d), this could be due to the window length of your model\r\n",
                  EI_CLASSIFIER_RAW_SAMPLE_COUNT);
        return;
    }
}

void loop() {
    /* ===== SECTION 1: CONTINUOUS AUDIO PROCESSING ===== */
    bool m = microphone_inference_record();
    if (!m) {
        ei_printf("ERR: Failed to record audio...\n");
        return;
    }

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
    signal.get_data = &microphone_audio_signal_get_data;

    ei_impulse_result_t result = { 0 };

    EI_IMPULSE_ERROR r = run_classifier_continuous(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", r);
        return;
    }

    /* ===== SECTION 2: RESULT PROCESSING (TOP-1) =====
       Find the label with highest confidence score in this inference cycle
    */
    float max_val = 0.0f;
    const char* best_label = "unknown";

    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        if (result.classification[ix].value > max_val) {
            max_val = result.classification[ix].value;
            best_label = result.classification[ix].label;
        }
    }

    /* Optional debug printing */
    if (++print_results >= (EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)) {
        ei_printf("Top: %s (%.2f)\n", best_label, max_val);
        print_results = 0;
    }

    /* ===== SECTION 3: STATE MACHINE + TIMEOUT + DEBOUNCE ===== */
    unsigned long current_time = millis();

    /* 1) Timeout handling: ACTIVE too long -> back to IDLE */
    if (current_state == STATE_ACTIVE &&
        (current_time - last_activity_time > COMMAND_TIMEOUT_MS)) {
        current_state = STATE_IDLE;
        set_rgb(true, false, false); // Đỏ: IDLE
        ei_printf("--- TIMEOUT: System Sleep ---\n");
    }

    /* 2) Process prediction only when confidence passes threshold */
    if (max_val > CONFIDENCE_THRESHOLD) {

        /* Ignore noise / unknown */
        if (strcmp(best_label, "_noise") == 0 || strcmp(best_label, "_unknown") == 0) {
            return;
        }

        switch (current_state) {
            case STATE_IDLE:
                /* IDLE: Only WAKE transitions to ACTIVE */
                if (strcmp(best_label, "WAKE") == 0) {
                    /* Debounce nhẹ cho WAKE */
                    if (current_time - last_action_executed_time > 1000) {
                        current_state = STATE_ACTIVE;
                        last_activity_time = current_time;
                        last_action_executed_time = current_time;
                        set_rgb(false, false, true); // Xanh dương: Listening
                        ei_printf(">>> WAKE UP DETECTED! Waiting for commands...\n");
                    }
                }
                break;

            case STATE_ACTIVE:
                /* ACTIVE: Listen for commands */
                last_activity_time = current_time;

                if (strcmp(best_label, "WAKE") == 0) {
                    /* If WAKE detected again: do nothing (only timeout reset above) */
                }
                else {
                    /* Debounce between executed actions */
                    if (current_time - last_action_executed_time > ACTION_DEBOUNCE_MS) {
                        execute_command(best_label);
                        last_action_executed_time = current_time;
                    }
                }
                break;
        }
    }
}

/* ========== HELPER FUNCTIONS (HARDWARE + COMMANDS) ========== */

/* Control RGB LED on Nano 33 BLE (active-low) */
static void set_rgb(bool r, bool g, bool b) {
    digitalWrite(LEDR, r ? LOW : HIGH);
    digitalWrite(LEDG, g ? LOW : HIGH);
    digitalWrite(LEDB, b ? LOW : HIGH);
}

static void fan_control(bool on) {
    if (on) {
        digitalWrite(FAN_IN1, HIGH);
        digitalWrite(FAN_IN2, LOW);
    } else {
        digitalWrite(FAN_IN1, LOW);
        digitalWrite(FAN_IN2, LOW);
    }
}

static void execute_command(const char* label) {
    bool action_taken = false;

    if (strcmp(label, "fan-on") == 0) {
        fan_control(true);
        ei_printf(">>> ACTION: FAN ON\n");
        action_taken = true;
    }
    else if (strcmp(label, "fan-off") == 0) {
        fan_control(false);
        ei_printf(">>> ACTION: FAN OFF\n");
        action_taken = true;
    }
    else if (strcmp(label, "led-on") == 0) {
        digitalWrite(LED_BUILTIN, HIGH);
        ei_printf(">>> ACTION: LED ON\n");
        action_taken = true;
    }
    else if (strcmp(label, "led-off") == 0) {
        digitalWrite(LED_BUILTIN, LOW);
        ei_printf(">>> ACTION: LED OFF\n");
        action_taken = true;
    }

    /* Effect: blink green to acknowledge successful command */
    if (action_taken) {
        set_rgb(false, true, false);
        delay(200);
        set_rgb(false, false, true); // Back to blue
    }
}

/* ========== AUDIO PROCESSING FUNCTIONS (From SDK) ========== */

static void pdm_data_ready_inference_callback(void) {
    int bytesAvailable = PDM.available();
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);

    if (record_ready == true) {
        for (int i = 0; i < bytesRead >> 1; i++) {
            inference.buffers[inference.buf_select][inference.buf_count++] = sampleBuffer[i];

            if (inference.buf_count >= inference.n_samples) {
                inference.buf_select ^= 1;
                inference.buf_count = 0;
                inference.buf_ready = 1;
            }
        }
    }
}

static bool microphone_inference_start(uint32_t n_samples) {
    inference.buffers[0] = (signed short *)malloc(n_samples * sizeof(signed short));
    if (inference.buffers[0] == NULL) return false;

    inference.buffers[1] = (signed short *)malloc(n_samples * sizeof(signed short));
    if (inference.buffers[1] == NULL) {
        free(inference.buffers[0]);
        return false;
    }

    sampleBuffer = (signed short *)malloc((n_samples >> 1) * sizeof(signed short));
    if (sampleBuffer == NULL) {
        free(inference.buffers[0]);
        free(inference.buffers[1]);
        return false;
    }

    inference.buf_select = 0;
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    PDM.onReceive(&pdm_data_ready_inference_callback);
    PDM.setBufferSize((n_samples >> 1) * sizeof(int16_t));

    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        ei_printf("Failed to start PDM!");
    }

    PDM.setGain(127);
    record_ready = true;

    return true;
}

static bool microphone_inference_record(void) {
    bool ret = true;

    if (inference.buf_ready == 1) {
        ei_printf(
            "Error sample buffer overrun. Decrease the number of slices per model window "
            "(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)\n");
        ret = false;
    }

    while (inference.buf_ready == 0) {
        delay(1);
    }

    inference.buf_ready = 0;
    return ret;
}

static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
    numpy::int16_to_float(&inference.buffers[inference.buf_select ^ 1][offset], out_ptr, length);
    return 0;
}

static void microphone_inference_end(void) {
    PDM.end();
    free(inference.buffers[0]);
    free(inference.buffers[1]);
    free(sampleBuffer);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif
