/* Voice-controlled IoT system using Edge Impulse speech recognition */
/* Supports wake word detection and command execution for LED and FAN control */

#include <Arduino.h>
#include <PDM.h>  /* Pulse Density Modulation microphone library */
#include <Speech_Recognition_inferencing.h>  /* Edge Impulse inference library */

#include <deque>  /* For command history queue */
#include <string>  /* For string operations */
#include <vector>  /* For detection results */

/* Enable quantization for filterbank to optimize memory usage */
#define EIDSP_QUANTIZE_FILTERBANK    1

/* Perform inference every 300ms for responsive detection */
#define INFERENCE_EVERY_MS           300
/* Minimum confidence threshold (70%) to accept a prediction */
#define PREDICTION_THRESHOLD         0.7f
/* Timeout period (5 seconds) - system returns to idle if no command received */
#define LISTENING_TIMEOUT_MS         5000

/* Circular buffer size matching the classifier's required sample count */
#define RING_BUFFER_SIZE             (EI_CLASSIFIER_RAW_SAMPLE_COUNT)

/* Hardware pin definitions */
#define BUZZER_PIN                   D2  /* Buzzer for audio feedback */
#define FAN_PIN                      D3  /* Fan control output */

/* Circular buffer for audio samples used by the classifier */
static int16_t ring_buffer[RING_BUFFER_SIZE];
/* Current write position in the ring buffer (volatile for ISR access) */
static volatile int write_index = 0;
/* Flag indicating if the buffer has wrapped around at least once */
static volatile bool buffer_filled_once = false;
/* Counter tracking samples collected since last inference */
static int samples_since_last_inference = 0;

/* Temporary buffer for PDM microphone data */
static short sampleBuffer[2048];

/* FSM States */
enum SystemState {
    STATE_IDLE,       /* Waiting for wake word */
    STATE_LISTENING,  /* Active listening for commands */
};

/* Current system state - starts in idle mode */
SystemState current_state = STATE_IDLE;
/* Timestamp of last wake word detection or command - used for timeout */
unsigned long last_wake_time = 0;

/* Structure to hold detected keyword information */
struct DetectedKeyword {
    String label;              /* Detected keyword label */
    unsigned long timestamp;   /* Time of detection */
    float confidence;          /* Confidence score (0.0 to 1.0) */
};

/* Structure to store command history for sequence analysis */
struct CommandHistory {
    String label;              /* Command label */
    unsigned long timestamp;   /* Time of detection */
};
/* Queue storing recent commands (max 5) for pattern matching */
std::deque<CommandHistory> active_history;

/* FSM processing function - handles state transitions and command execution */
void process_fsm(std::vector<DetectedKeyword> detections);
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
    while (!Serial);  /* Wait for serial port to connect */

    Serial.println("Edge Impulse Wake Word Demo (Continuous)");

    /* Configure all output pins */
    pinMode(LED_BUILTIN, OUTPUT);  /* Built-in LED for user control */
    pinMode(LEDR, OUTPUT);         /* Red LED for status indication */
    pinMode(LEDG, OUTPUT);         /* Green LED for status indication */
    pinMode(LEDB, OUTPUT);         /* Blue LED for status indication */
    pinMode(BUZZER_PIN, OUTPUT);   /* Buzzer for audio feedback */
    pinMode(FAN_PIN, OUTPUT);      /* Fan control output */

    /* Set initial state - Red LED indicates idle/sleeping state */
    RGB_control(true, false, false);
    digitalWrite(BUZZER_PIN, LOW);  /* Ensure buzzer is off */

    /* Display classifier configuration for debugging */
    ei_printf("Inferencing settings:\n");
    ei_printf("\tInterval: %.2f ms.\n", (float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);

    /* Configure PDM microphone */
    PDM.onReceive(&pdm_data_ready_inference_callback);  /* Set ISR callback */
    PDM.setBufferSize(2048);  /* Set buffer size for audio samples */
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {  /* Start PDM with classifier frequency */
        ei_printf("Failed to start PDM!");
        while (1);  /* Halt execution on failure */
    }
    PDM.begin(1, 16000);  /* Mono channel, 16kHz sampling rate */
    PDM.setGain(127);     /* Set microphone gain to maximum */
}

/* Main loop - runs continuously performing inference at regular intervals */
void loop() {
    /* Calculate required samples for the desired inference interval */
    int samples_to_wait = (INFERENCE_EVERY_MS * EI_CLASSIFIER_FREQUENCY) / 1000;

    /* Check if enough samples have been collected for next inference */
    if (samples_since_last_inference >= samples_to_wait) {
        samples_since_last_inference = 0;  /* Reset counter for next inference */

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
            return 0;  /* Success */
        };

        /* Initialize result structure and run classifier */
        ei_impulse_result_t result = {0};
        EI_IMPULSE_ERROR r = run_classifier(&signal, &result, false);
        if (r != EI_IMPULSE_OK) return;  /* Exit if classifier fails */

        /* Find the classification with highest confidence */
        float max_val = 0;
        const char* max_lbl = "_unknown";

        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            if (result.classification[ix].value > max_val) {
                max_val = result.classification[ix].value;
                max_lbl = result.classification[ix].label;
            }
        }

        /* Vector to store valid detections */
        std::vector<DetectedKeyword> current_detection;
        ei_printf("Debug: %s = %.2f\n", max_lbl, max_val);

        /* Filter out low confidence predictions and noise/unknown labels */
        if (max_val > PREDICTION_THRESHOLD && strcmp(max_lbl, "_unknown") != 0 && strcmp(max_lbl, "_noise") != 0 &&
            strcmp(max_lbl, "unknown") != 0 && strcmp(max_lbl, "noise") != 0) {
            /* Add valid detection to vector with timestamp */
            current_detection.push_back({String(max_lbl), millis(), max_val});
            ei_printf("Detected: %s (%.2f)\n", max_lbl, max_val);
        }

        /* Process detection through finite state machine */
        process_fsm(current_detection);
    }
}

/* Finite State Machine processor - handles wake word and command logic */
void process_fsm(std::vector<DetectedKeyword> detections) {
    unsigned long current_time = millis();

    /* Check timeout - if listening state but no command for too long, go to sleep */
    if (current_state == STATE_LISTENING) {
        if (current_time - last_wake_time > LISTENING_TIMEOUT_MS) {
            /* Transition back to idle state */
            current_state = STATE_IDLE;
            active_history.clear();  /* Clear command history */

            /* Red LED: System sleeping */
            RGB_control(true, false, false);
            ei_printf("--- TIMEOUT: System Sleep (Waiting for 'hello') ---\n");
        }
    }

    /* Exit if no detections to process */
    if (detections.empty()) return;
    DetectedKeyword d = detections[0];  /* Get the detected keyword */

    /* Process FSM based on current state */
    switch (current_state) {
        /* IDLE State - waiting for wake word */
        case STATE_IDLE:
            /* Only respond to WAKE keyword */
            if (d.label == "WAKE") {
                /* Transition to listening state */
                current_state = STATE_LISTENING;
                last_wake_time = current_time;  /* Record wake time for timeout */

                /* Blue LED: Awake and listening */
                RGB_control(false, false, true);

                /* Provide audio feedback */
                digitalWrite(BUZZER_PIN, HIGH);
                delay(500);
                digitalWrite(BUZZER_PIN, LOW);
                ei_printf(">>> WAKE WORD DETECTED! ('hello') -> Listening...\n");
            }
            break;

        /* LISTENING State - processing commands */
        case STATE_LISTENING:
            /* Update timeout timer on each valid detection */
            last_wake_time = current_time;

            /* Add keyword to history buffer (maintains last 5 commands) */
            active_history.push_back({d.label, d.timestamp});
            if (active_history.size() > 5) active_history.pop_front();  /* Remove oldest */

            /* Parse command sequence from history */
            String action = "";  /* Will be ON or OFF */
            String device = "";  /* Will be LED or FAN */

            /* Analyze recent history to find action-device pairs */
            for (auto& h : active_history) {
                /* Only consider keywords from last 2 seconds */
                if (current_time - h.timestamp > 2000) continue;

                /* Identify action keywords */
                if (h.label == "ON" || h.label == "OFF")
                    action = h.label;
                /* Identify device keywords */
                else if (h.label == "LED" || h.label == "FAN")
                    device = h.label;
            }

            /* If we have both action and device, execute command */
            if (action != "" && device != "") {
                ei_printf(">>> COMMAND MATCHED: %s %s <<<\n", device.c_str(), action.c_str());

                /* Green LED: Command executed successfully */
                RGB_control(false, true, false);

                /* Execute hardware control based on parsed command */
                if (device == "LED") {
                    /* Control built-in LED */
                    digitalWrite(LED_BUILTIN, (action == "ON") ? HIGH : LOW);
                } else if (device == "FAN") {
                    /* Control fan motor */
                    fan_control(action == "ON");
                }

                /* Clear history after execution to prevent command repeat */
                active_history.clear();

                /* Stay in listening state to accept more commands */
                delay(500);  /* Brief pause for visual feedback */
                RGB_control(false, false, true);  /* Return to blue LED */
            }
            break;
    }
}

/* Fan control function - turns fan motor on or off */
void fan_control(bool on) {
    digitalWrite(FAN_PIN, on ? HIGH : LOW);  /* Set fan pin state */
}

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
        write_index++;  /* Advance write position */
        /* Handle circular buffer wraparound */
        if (write_index >= RING_BUFFER_SIZE) {
            write_index = 0;  /* Wrap to beginning */
            buffer_filled_once = true;  /* Mark buffer as having valid data */
        }
    }
    /* Update sample counter for inference timing */
    samples_since_last_inference += samplesRead;
}

/* RGB LED control function - manages status indication LEDs */
/* Note: LEDs are active-low (LOW = ON, HIGH = OFF) */
void RGB_control(bool red, bool green, bool blue) {
    digitalWrite(LEDR, red ? LOW : HIGH);    /* Control red LED */
    digitalWrite(LEDG, green ? LOW : HIGH);  /* Control green LED */
    digitalWrite(LEDB, blue ? LOW : HIGH);   /* Control blue LED */
}
