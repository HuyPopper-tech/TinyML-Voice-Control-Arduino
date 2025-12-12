/* Voice-controlled IoT system using Edge Impulse speech recognition */
/* Supports wake word detection and command execution for LED and FAN control */

#include <Arduino.h>
#include <PDM.h>
#include <Speech_Recognition_inferencing.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LABEL_LEN                           7

/* Enable quantization for filterbank to optimize memory usage */
#define EIDSP_QUANTIZE_FILTERBANK               1

/* Perform inference every 200ms for responsive detection */
#define INFERENCE_EVERY_MS                      200
/* Minimum confidence threshold (70%) to accept a prediction */
#define PREDICTION_THRESHOLD                    0.7f
/* Timeout period (10 seconds) - system returns to idle if no command received */
#define LISTENING_TIMEOUT_MS                    10000
#define DEVICE_TIMEOUT_MS                       4000

/* Circular buffer size matching the classifier's required sample count */
#define RING_BUFFER_SIZE                        (3 * EI_CLASSIFIER_RAW_SAMPLE_COUNT)

/* Hardware pins */
#define BUZZER_PIN                    D2
#define FAN_PIN                       D3

/* Circular buffer for audio samples used by the classifier */
static int16_t ring_buffer[RING_BUFFER_SIZE];
static volatile int write_index = 0;
/* Flag indicating if the buffer has wrapped around at least once */
static volatile bool buffer_filled_once = false;
/* Counter tracking samples collected since last inference */
static int samples_since_last_inference = 0;

/* Temporary buffer for PDM microphone data */
static short sampleBuffer[2048];

/* Finite-state machine states */
enum SystemState : uint8_t {
    STATE_IDLE,
    STATE_WAIT_ACTION,
    STATE_DEVICE_ON,
    STATE_DEVICE_OFF
};

/* Current system state - starts in idle mode */
SystemState current_state = STATE_IDLE;
/* Timestamp of last wake word detection or command - used for timeout */
static unsigned long last_wake_time = 0;
static unsigned long last_device_time = 0;
static char last_label_processed[MAX_LABEL_LEN] = {0};
static unsigned long last_label_time = 0;

/* FSM processing function - handles state transitions and command execution */
void process_fsm(const char* label, float confidence);
/* Controls the RGB LED with specified colors (true = on, false = off) */
void RGB_control(bool red, bool green, bool blue);
/* Controls the fan motor (true = on, false = off) */
void fan_control(bool on);
/* ISR callback for PDM microphone - fills ring buffer with audio samples */
static void pdm_data_ready_inference_callback(void);

/* System initialization - runs once at startup */
void setup() {
    /* Initialize serial communication at high baud rate for debugging */
    Serial.begin(921600);
    while (!Serial); /* Wait for serial port to connect */

    Serial.println("Edge Impulse Wake Word Demo (Continuous)");

    /* Configure all output pins */
    pinMode(LED_BUILTIN, OUTPUT); /* Built-in LED for user control */
    pinMode(LEDR, OUTPUT);        /* Red LED for status indication */
    pinMode(LEDG, OUTPUT);        /* Green LED for status indication */
    pinMode(LEDB, OUTPUT);        /* Blue LED for status indication */
    pinMode(BUZZER_PIN, OUTPUT);  /* Buzzer for audio feedback */
    pinMode(FAN_PIN, OUTPUT);     /* Fan control output */

    /* Set initial state - Red LED indicates idle/sleeping state */
    RGB_control(true, false, false);
    digitalWrite(BUZZER_PIN, LOW); /* Ensure buzzer is off */

    /* Display classifier configuration for debugging */
    ei_printf("Inferencing settings:\n");
    ei_printf("\tInterval: %.2f ms.\n", (float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);

    /* Configure PDM microphone */
    PDM.onReceive(&pdm_data_ready_inference_callback); /* Set ISR callback */
    PDM.setBufferSize(2048);                           /* Set buffer size for audio samples */
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {      /* Start PDM with classifier frequency */
        ei_printf("Failed to start PDM!");
        while (1); /* Halt execution on failure */
    }
    PDM.begin(1, 16000); /* Mono channel, 16kHz sampling rate */
    PDM.setGain(127);    /* Set microphone gain to maximum */
}

/* Main loop - runs continuously performing inference at regular intervals */
void loop() {
    /* Calculate required samples for the desired inference interval */
    int samples_to_wait = (INFERENCE_EVERY_MS * EI_CLASSIFIER_FREQUENCY) / 1000;

    /* Check if enough samples have been collected for next inference */
    if (samples_since_last_inference >= samples_to_wait) {
        samples_since_last_inference = 0; /* Reset counter for next inference */

        /* Wait until we have enough data in the buffer before first inference */
        if (!buffer_filled_once && write_index < EI_CLASSIFIER_RAW_SAMPLE_COUNT) {
            return;
        }

        /* Prepare signal structure for classifier */
        signal_t signal;
        signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;

        /* Lambda function to read data from circular buffer */
        signal.get_data = [](size_t offset, size_t length, float* out_ptr) -> int {
            /* Calculate starting read position (most recent samples) */
            int read_start_index = write_index - EI_CLASSIFIER_RAW_SAMPLE_COUNT + offset;

            /* Handle circular buffer wraparound */
            if (read_start_index < 0)
                read_start_index += RING_BUFFER_SIZE;
            else if (read_start_index >= RING_BUFFER_SIZE)
                read_start_index -= RING_BUFFER_SIZE;

            /* Copy samples from ring buffer to output, converting to float */
            for (size_t i = 0; i < length; i++) {
                int idx = (read_start_index + i) % RING_BUFFER_SIZE;
                out_ptr[i] = (float)ring_buffer[idx];
            }
            return 0; /* Success */
        };

        /* Initialize result structure and run classifier */
        ei_impulse_result_t result = {0};
        EI_IMPULSE_ERROR r = run_classifier(&signal, &result, false);
        if (r != EI_IMPULSE_OK) return; /* Exit if classifier fails */

        /* Find the classification with highest confidence */
        float max_val = 0;
        const char* max_lbl = "_unknown";

        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            if (result.classification[ix].value > max_val) {
                max_val = result.classification[ix].value;
                max_lbl = result.classification[ix].label;
            }
        }

        ei_printf("Debug: %s = %.2f\n", max_lbl, max_val);

        /* Filter out low confidence predictions and noise/unknown labels */
        if (max_val > PREDICTION_THRESHOLD && strcmp(max_lbl, "_noise") != 0 && strcmp(max_lbl, "noise") != 0) {
            ei_printf("Detected: %s (%.2f)\n", max_lbl, max_val);
            process_fsm(max_lbl, max_val);
        }
    }
}

/* Finite State Machine processor - handles wake word and command logic */
void process_fsm(const char* label, float confidence) {
    unsigned long current_time = millis();

    /* Step 1: If listening timed out, go back to idle */
    if (current_state != STATE_IDLE && (current_time - last_wake_time > LISTENING_TIMEOUT_MS)) {
        current_state = STATE_IDLE;
        RGB_control(true, false, false);
        ei_printf("--- Timeout: He thong di ngu ---\n");
    }

    /* Step 2: Ignore noise/unknown and suppress repeats (<500 ms) */
    if (label == "_noise" || label == "_unknown") return;

    /* Debounce: if repeated label within 500ms, ignore duplicates */
    if (strncmp(label, last_label_processed, MAX_LABEL_LEN) == 0 && (current_time - last_label_time < 500)) {
        return;
    }

    strncpy(last_label_processed, label, MAX_LABEL_LEN - 1);
    last_label_processed[MAX_LABEL_LEN - 1] = '\0';
    last_label_time = current_time;

    /* Step 3: Handle states and actions */

    switch (current_state) {
        case STATE_IDLE:
            if (strcmp(label, "WAKE") == 0) {
                current_state = STATE_WAIT_ACTION;
                last_wake_time = current_time;
                RGB_control(false, false, true);
                digitalWrite(BUZZER_PIN, HIGH);
                delay(200);
                digitalWrite(BUZZER_PIN, LOW);

                ei_printf(">>> DA THUC! Cho lenh trong 10 giay...\n");
            }
            break;

        case STATE_WAIT_ACTION:
            last_wake_time = current_time;

            if (strcmp(label, "WAKE") == 0) {
                ei_printf("... (Van dang nghe) ...\n");
            } else if (strcmp(label, "ON") == 0) {
                current_state = STATE_DEVICE_ON;
                ei_printf("[FSM] ACTION = ON, cho thiet bi...\n");
            } else if (strcmp(label, "OFF") == 0) {
                current_state = STATE_DEVICE_OFF;
                ei_printf("[FSM] ACTION = OFF, cho thiet bi...\n");
            }
            break;

        case STATE_DEVICE_ON:
            if (current_time - last_device_time > DEVICE_TIMEOUT_MS) {
                current_state = STATE_WAIT_ACTION;
                last_device_time = current_time;
                ei_printf("--- Timeout 2s: Reset ve WAIT_ACTION ---\n");
            }
            last_wake_time = current_time;

            if (strcmp(label, "WAKE") == 0) {
                /* keep state */
            } else if (strcmp(label, "ON") == 0) {
                /* keep ON */
            } else if (strcmp(label, "OFF") == 0) {
                current_state = STATE_DEVICE_OFF;
                ei_printf("[FSM] Doi ACTION -> OFF\n");
            } else if (strcmp(label, "LED") == 0) {
                ei_printf(">>> THUC THI: LED ON <<<\n");
                RGB_control(false, true, false);
                digitalWrite(LED_BUILTIN, HIGH);

                delay(500);
                RGB_control(false, false, true);
                current_state = STATE_WAIT_ACTION;
            } else if (strcmp(label, "FAN") == 0) {
                ei_printf(">>> THUC THI: FAN ON <<<\n");
                RGB_control(false, true, false);
                fan_control(true);

                delay(500);
                RGB_control(false, false, true);
                current_state = STATE_WAIT_ACTION;
            }
            break;

        case STATE_DEVICE_OFF:
            if (current_time - last_device_time > DEVICE_TIMEOUT_MS) {
                current_state = STATE_WAIT_ACTION;
                last_device_time = current_time;
                ei_printf("--- Timeout 2s: Reset ve WAIT_ACTION ---\n");
            }
            last_wake_time = current_time;

            if (strcmp(label, "WAKE") == 0) {
                /* keep state */
            } else if (strcmp(label, "OFF") == 0) {
                /* keep OFF */
            } else if (strcmp(label, "ON") == 0) {
                current_state = STATE_DEVICE_ON;
                ei_printf("[FSM] Doi ACTION -> ON\n");
            } else if (strcmp(label, "LED") == 0) {
                ei_printf(">>> THUC THI: LED OFF <<<\n");
                RGB_control(false, true, false);
                digitalWrite(LED_BUILTIN, LOW);

                delay(500);
                RGB_control(false, false, true);
                current_state = STATE_WAIT_ACTION;
            } else if (strcmp(label, "FAN") == 0) {
                ei_printf(">>> THUC THI: FAN OFF <<<\n");
                RGB_control(false, true, false);
                fan_control(false);

                delay(500);
                RGB_control(false, false, true);
                current_state = STATE_WAIT_ACTION;
            }
            break;
    }
}

/* Fan control function - turns fan motor on or off */
void fan_control(bool on) { digitalWrite(FAN_PIN, on ? HIGH : LOW); /* Set fan pin state */ }

/* ISR callback function - called when PDM microphone has new data available */
static void pdm_data_ready_inference_callback(void) {
    /* Check how many bytes are available from the microphone */
    int bytesAvailable = PDM.available();
    /* Read audio data into temporary buffer */
    int bytesRead = PDM.read((char*)&sampleBuffer[0], bytesAvailable);
    /* Calculate number of samples (each sample is 2 bytes/16-bit) */
    int samplesRead = bytesRead / 2;

    /* Transfer samples from temporary buffer to circular ring buffer */
    for (int i = 0; i < samplesRead; i++) {
        ring_buffer[write_index] = sampleBuffer[i];
        write_index++; /* Advance write position */
        /* Handle circular buffer wraparound */
        if (write_index >= RING_BUFFER_SIZE) {
            write_index = 0;           /* Wrap to beginning */
            buffer_filled_once = true; /* Mark buffer as having valid data */
        }
    }
    /* Update sample counter for inference timing */
    samples_since_last_inference += samplesRead;
}

/* RGB LED control function - manages status indication LEDs */
/* Note: LEDs are active-low (LOW = ON, HIGH = OFF) */
void RGB_control(bool red, bool green, bool blue) {
    digitalWrite(LEDR, red ? LOW : HIGH);   /* Control red LED */
    digitalWrite(LEDG, green ? LOW : HIGH); /* Control green LED */
    digitalWrite(LEDB, blue ? LOW : HIGH);  /* Control blue LED */
}
