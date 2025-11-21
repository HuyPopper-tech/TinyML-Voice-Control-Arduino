/*
 ** NOTE: If you run into TFLite arena allocation issue.
 ** Try defining "-DEI_CLASSIFIER_ALLOCATION_STATIC" in boards.local.txt
 */

/* Includes ---------------------------------------------------------------- */
#include <PDM.h>
#include <Arduino.h>
#include <Speech_Recognition_inferencing.h>
#include <vector>
#include <string>
#include <cmath>

/* Memory optimization macro */
// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK 1

/* Audio buffer settings */
#define NUM_FRAMES                 3
#define FRAME_SAMPLES              EI_CLASSIFIER_RAW_SAMPLE_COUNT
#define TOTAL_SAMPLES              (NUM_FRAMES * FRAME_SAMPLES)

/* Hardware Pin Definitions */
#define LEDR                       22
#define LEDG                       23
#define LEDB                       24

#define L298N_IN1                  8
#define L298N_IN2                  9
#define L298N_ENA                  10

/* Sliding Window Configuration */
#define WINDOW_STEP_MS             300    /* Step size for the sliding window (300ms stride) */
#define PREDICTION_THRESHOLD       0.7f  /* Minimum confidence to accept a prediction */

/** Audio buffers, pointers and selectors */
typedef struct {
    int16_t *buffer;
    uint8_t buf_ready;
    uint32_t buf_count;
    uint32_t n_samples;
} inference_t;

/* Structure to hold detected keywords with timestamp */
struct DetectedKeyword {
    String label;
    unsigned long timestamp; 
    float confidence;
};

/* Global variables */
static inference_t inference;
static signed short sampleBuffer[2048];   /* Temporary buffer for PDM callback */
static bool debug_nn = false;             /* Set true to debug features */
static int window_offset = 0;             /* Tracks current position in the sliding window */

/* Function Prototypes */
static void pdm_data_ready_inference_callback(void);
static bool microphone_inference_start(uint32_t n_samples);
static bool microphone_inference_record(void);
static void microphone_inference_end(void);
void process_command(std::vector<DetectedKeyword> detections);
void RGB_control(bool red, bool green, bool blue); 
void motor_control(bool forward, bool backward, int speed);

/**
 * @brief      Arduino setup function
 */
void setup() {
    Serial.begin(115200);
    // while (!Serial); /* Optional: Wait for Serial Monitor */
    
    Serial.println("Edge Impulse Inferencing Demo");

    /* Configure LED pins */
    pinMode(LED_BUILTIN, OUTPUT);
    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);

    pinMode(L298N_IN1, OUTPUT);
    pinMode(L298N_IN2, OUTPUT);
    pinMode(L298N_ENA, OUTPUT);

    /* Print model settings */
    ei_printf("Inferencing settings:\n");
    ei_printf("\tInterval: %.2f ms.\n", (float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    ei_printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0]));

    /* Initialize microphone buffer */
    if (microphone_inference_start(TOTAL_SAMPLES) == false) {
        ei_printf("ERR: Could not allocate audio buffer (size %d)\r\n", TOTAL_SAMPLES);
        return;
    }

    RGB_control(false, false, false);
}

/**
 * @brief      Arduino main function. Runs the inferencing loop.
 */
void loop() {
    ei_printf("Starting inferencing in 2 seconds...\n");
    delay(2000);

    ei_printf("Recording...\n");

    /* Record audio into the big buffer */
    bool m = microphone_inference_record();
    if (!m) {
        ei_printf("ERR: Failed to record audio...\n");
        return;
    }

    ei_printf("Recording done\n");

    /* Calculate sliding window parameters */
    int window_samples = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    int stride_samples = (WINDOW_STEP_MS * EI_CLASSIFIER_FREQUENCY) / 1000;
    
    std::vector<DetectedKeyword> raw_detections;

    /* * Sliding Window Algorithm:
     * Iterate through the total recorded buffer (3s) in small steps (stride).
     * This ensures keywords split between seconds are captured correctly.
     */
    for (int offset = 0; offset <= TOTAL_SAMPLES - window_samples; offset += stride_samples) {
        window_offset = offset; /* Update global offset for the lambda function */

        signal_t signal;
        signal.total_length = window_samples;
        
        /* * Lambda function to feed data to the classifier.
         * It converts int16 data to float on-the-fly to save RAM.
         */
        signal.get_data = [](size_t off, size_t len, float *out_ptr) -> int {
            /* Calculate actual index in the main buffer */
            int idx_in_big_buffer = window_offset + off;
            
            /* Convert only the necessary chunk from int16 to float */
            numpy::int16_to_float(&inference.buffer[idx_in_big_buffer], out_ptr, len);
            
            return 0;
        };

        ei_impulse_result_t result = {0};

        /* Run inference on the current window slice */
        EI_IMPULSE_ERROR r = run_classifier(&signal, &result, false);

        if (r != EI_IMPULSE_OK) {
            ei_printf("ERR: Failed to run classifier (%d)\n", r);
            break;
        }

        /* Find the label with the highest confidence */
        float max_value = 0;
        const char *max_label = "unknown";
        
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            if (result.classification[ix].value > max_value) {
                max_value = result.classification[ix].value;
                max_label = result.classification[ix].label;
            }
        }

        /* Store valid detections */
        if (max_value > PREDICTION_THRESHOLD && strcmp(max_label, "_unknown") != 0 && strcmp(max_label, "_noise") != 0) {
            /* Calculate timestamp relative to the start of recording */
            unsigned long timestamp_ms = (offset * 1000) / EI_CLASSIFIER_FREQUENCY;
            
            raw_detections.push_back({String(max_label), timestamp_ms, max_value});
        }
    }

    /* Process the list of raw detections to form commands */
    process_command(raw_detections);
}

/**
 * @brief      Process detected commands with logic (Debouncing & Semantic Parsing)
 * @param[in]  detections  The list of raw detections from the sliding window
 */
void process_command(std::vector<DetectedKeyword> detections) {
    if (detections.empty()) {
        ei_printf("No valid commands detected.\n");
        return;
    }

    /* Step 1: Debouncing / filtering duplicates */
    std::vector<DetectedKeyword> unique_cmds;

    for (auto &d : detections) {
        bool merged = false;
        if (!unique_cmds.empty()) {
            DetectedKeyword &last = unique_cmds.back();

            /* If the same label appears within 450ms, consider it the same word */
            if (d.label == last.label && (d.timestamp - last.timestamp) < 450) {
                if (d.confidence > last.confidence) {
                    last = d; /* Keep the one with higher confidence */
                }
                merged = true;
            }
        }
        if (!merged) {
            unique_cmds.push_back(d);
        }
    }

    /* Print the filtered sequence */
    String sequence_str = "";
    for (auto &d : unique_cmds) {
        ei_printf("  > %s (at %dms, conf: %.2f)\n", d.label.c_str(), d.timestamp, d.confidence);
        sequence_str += d.label + " ";
    }

    /* Step 2: Semantic Parsing (Action + Device) */
    String action = "";
    String device = "";
    long action_time = -1;
    long device_time = -1;

    for (auto &d : unique_cmds) {
        if (d.label == "ON" || d.label == "OFF") {
            action = d.label;
            action_time = d.timestamp;
        } 
        else if (d.label == "TURN") {
            /* "TURN" is a trigger word, usually followed by ON/OFF or implies ON */
            if (action == "") {
                 /* Logic for standalone TURN can be added here */
            }
        }
        else if (d.label == "LED" || d.label == "FAN") {
            device = d.label;
            device_time = d.timestamp;
        }
    }

    /* Step 3: Execute Command if logic is satisfied */
    if (device != "" && action != "") {
        /* Check if Action and Device were spoken close to each other (within 2.5s) */
        if (abs(device_time - action_time) < 2500) {
            ei_printf("=> COMMAND MATCHED: %s %s\n", device.c_str(), action.c_str());
            
            /* Actuator Logic */
            if (device == "LED") {
                if (action == "ON") {
                    digitalWrite(LED_BUILTIN, HIGH);
                    ei_printf("   [ACTUATOR] LED turned ON\n");
                } else {
                    digitalWrite(LED_BUILTIN, LOW);
                    ei_printf("   [ACTUATOR] LED turned OFF\n");
                }
            } 
            else if (device == "FAN") {
                if (action == "ON") {
                    motor_control(true, false, 200); /* Forward at speed 200 */
                    RGB_control(false, true, false); /* Indicate FAN ON with GREEN LED */
                    ei_printf("   [ACTUATOR] FAN turned ON\n");
                } else {
                    motor_control(false, true, 0); /* Stop */
                    RGB_control(true, false, false); /* Indicate FAN OFF with RED LED */
                    ei_printf("   [ACTUATOR] FAN turned OFF\n");
                }
            }
        } else {
            ei_printf("=> IGNORED: Device and Action too far apart in time.\n");
        }
    } else {
        ei_printf("=> INCOMPLETE: Missing Device or Action.\n");
    }
}

void motor_control(bool forward, bool backward, int speed) {
    if (forward) {
        digitalWrite(L298N_IN1, HIGH);
        digitalWrite(L298N_IN2, LOW);
    } else if (backward) {
        digitalWrite(L298N_IN1, LOW);
        digitalWrite(L298N_IN2, HIGH);
    } else {
        digitalWrite(L298N_IN1, LOW);
        digitalWrite(L298N_IN2, LOW);
    }
    analogWrite(L298N_ENA, speed);
}

/**
 * @brief      Helper to control RGB LED
 */
void RGB_control(bool red, bool green, bool blue) {
    digitalWrite(LEDR, red ? HIGH : LOW);
    digitalWrite(LEDG, green ? HIGH : LOW);
    digitalWrite(LEDB, blue ? HIGH : LOW);
}

/**
 * @brief      PDM buffer full callback.
 * Reads data from PDM hardware into the intermediate buffer.
 */
static void pdm_data_ready_inference_callback(void) {
    int bytesAvailable = PDM.available();

    /* Read into the sample buffer */
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);

    if (inference.buf_ready == 0) {
        for (int i = 0; i < bytesRead >> 1; i++) {
            inference.buffer[inference.buf_count++] = sampleBuffer[i];

            if (inference.buf_count >= inference.n_samples) {
                inference.buf_ready = 1;
                break;
            }
        }
    }
}

/**
 * @brief      Initialize PDM microphone and allocate buffers.
 */
static bool microphone_inference_start(uint32_t n_samples) {
    inference.buffer = (int16_t *)malloc(n_samples * sizeof(int16_t));

    if (inference.buffer == NULL) {
        return false;
    }

    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    /* Configure PDM callback */
    PDM.onReceive(&pdm_data_ready_inference_callback);
    PDM.setBufferSize(4096);

    /* Initialize PDM: Mono, 16kHz */
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        ei_printf("Failed to start PDM!");
        microphone_inference_end();
        return false;
    }

    PDM.setGain(127);
    return true;
}

/**
 * @brief      Wait for PDM data to fill the buffer.
 */
static bool microphone_inference_record(void) {
    inference.buf_ready = 0;
    inference.buf_count = 0;

    while (inference.buf_ready == 0) {
        delay(10);
    }

    return true;
}

/**
 * @brief      Stop PDM and release memory.
 */
static void microphone_inference_end(void) {
    PDM.end();
    free(inference.buffer);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif