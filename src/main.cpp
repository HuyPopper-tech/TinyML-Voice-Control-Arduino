/* ============================================================================
 * Voice-controlled IoT System using Edge Impulse Speech Recognition
 * ============================================================================ */

#include <PDM.h>                            /* Pulse Density Modulation microphone library */
#include <Speech_Recognition_inferencing.h> /* Edge Impulse inference library */

/* ============================================================================
 * CONFIGURATION CONSTANTS
 * ============================================================================ */

/* Enable quantization for filterbank to optimize memory usage on Cortex-M4 */
#define EIDSP_QUANTIZE_FILTERBANK 1

/* Inference timing - perform inference every 100ms for responsive detection */
#define INFERENCE_EVERY_MS 100

/* Minimum confidence threshold (80%) to accept a single-frame prediction */
#define PREDICTION_THRESHOLD 0.8f

/* Timeout period (5 seconds) - system returns to idle if no command received */
#define LISTENING_TIMEOUT_MS 5000

/* ============================================================================
 * VOICE ACTIVITY DETECTION (VAD) PARAMETERS
 * Combined Energy + Zero-Crossing Rate for improved accuracy
 * ============================================================================ */

#define VAD_ENERGY_THRESHOLD 1000    /* Minimum energy level to consider as speech */
#define VAD_ZCR_SPEECH_MAX 0.15f     /* Maximum ZCR for voiced speech (lower = more voiced) */
#define VAD_ZCR_NOISE_MIN 0.30f      /* Minimum ZCR considered as noise/unvoiced */
#define VAD_ENERGY_WEIGHT 0.7f       /* Weight for energy component in combined VAD */
#define VAD_ZCR_WEIGHT 0.3f          /* Weight for ZCR component in combined VAD */
#define VAD_FRAMES_REQUIRED 3        /* Consecutive frames needed to confirm voice onset */
#define VAD_SILENCE_FRAMES_TIMEOUT 8 /* Consecutive silent frames to confirm voice offset */
#define VAD_HYSTERESIS_FACTOR 0.7f   /* Hysteresis factor to prevent rapid state switching */

/* ============================================================================
 * DUPLICATE DETECTION AND DEBOUNCING PARAMETERS
 * ============================================================================ */

#define SAME_LABEL_COOLDOWN_MS 400 /* Minimum time between same label detections */
#define MIN_CONFIDENCE_DIFF 0.15f  /* Confidence increase required to override cooldown */
#define MIN_WORDS_INTERVAL_MS 150  /* Minimum interval between different words */

/* ============================================================================
 * COMMAND PARSING PARAMETERS
 * ============================================================================ */

#define COMMAND_WINDOW_MS 1500 /* Time window to collect device + action pair */
#define COMMAND_HISTORY_SIZE 5 /* Maximum keywords stored in history buffer */

/* ============================================================================
 * POSTERIOR SMOOTHING AND CONFIDENCE ACCUMULATION
 * These techniques reduce false positives and missed detections
 * ============================================================================ */

#define SMOOTHING_WINDOW 4          /* Number of frames for moving average smoothing */
#define ACCUMULATION_THRESHOLD 1.2f /* Accumulated score threshold to accept detection */
#define ACCUMULATION_DECAY 0.75f    /* Decay factor for older frames in accumulation */

/* ============================================================================
 * BUFFER SIZES
 * ============================================================================ */

#define RING_BUFFER_SIZE (EI_CLASSIFIER_RAW_SAMPLE_COUNT)
#define VAD_SAMPLE_SIZE 256  /* Number of samples for VAD analysis */
#define PDM_BUFFER_SIZE 2048 /* PDM microphone buffer size */

/* ============================================================================
 * HARDWARE PIN DEFINITIONS
 * ============================================================================ */

#define BUZZER_PIN D2 /* Buzzer for audio feedback */
#define FAN_PIN D3    /* Fan control output */

/* ============================================================================
 * KEYWORD LABEL ENUMERATION
 * Using enum instead of String to avoid heap fragmentation on embedded systems
 * ============================================================================ */

typedef enum {
    LABEL_UNKNOWN = 0,
    LABEL_NOISE,
    LABEL_WAKE,
    LABEL_ON,
    LABEL_OFF,
    LABEL_LED,
    LABEL_FAN,
    LABEL_COUNT
} KeywordLabel;

/* Lookup table to convert label string to enum */
static const char* LABEL_STRINGS[] = {"_unknown", "_noise", "WAKE", "ON", "OFF", "LED", "FAN"};

/* ============================================================================
 * SYSTEM STATE MACHINE
 * ============================================================================ */

typedef enum {
    STATE_IDLE,     /* Waiting for wake word - low power mode */
    STATE_LISTENING /* Active listening for device + action commands */
} SystemState;

/* ============================================================================
 * DATA STRUCTURES (Static allocation to avoid heap fragmentation)
 * ============================================================================ */

/* Structure to store detected keyword information */
typedef struct {
    KeywordLabel label;      /* Detected keyword label (enum) */
    unsigned long timestamp; /* Time of detection in milliseconds */
    float confidence;        /* Confidence score (0.0 to 1.0) */
} DetectedKeyword;

/* Structure for command history - static circular buffer */
typedef struct {
    DetectedKeyword items[COMMAND_HISTORY_SIZE];
    uint8_t head;  /* Index of oldest item */
    uint8_t count; /* Number of items in buffer */
} CommandHistory;

/* Structure for posterior smoothing history */
typedef struct {
    float values[EI_CLASSIFIER_LABEL_COUNT];
    unsigned long timestamp;
} PosteriorFrame;

/* Structure for confidence accumulation across multiple frames */
typedef struct {
    float scores[EI_CLASSIFIER_LABEL_COUNT];
} AccumulatedConfidence;

/* ============================================================================
 * GLOBAL VARIABLES
 * ============================================================================ */

/* Audio ring buffer for continuous sampling */
static int16_t ring_buffer[RING_BUFFER_SIZE];
static volatile int write_index = 0;
static volatile bool buffer_filled_once = false;
static int samples_since_last_inference = 0;

/* Temporary buffer for PDM microphone data */
static int16_t pdm_sample_buffer[PDM_BUFFER_SIZE];

/* Voice Activity Detection state */
static int vad_active_frames = 0;
static int vad_silence_frames = 0;
static bool voice_detected = false;
static float adaptive_vad_threshold = 0.3f;

/* System state machine */
static SystemState current_state = STATE_IDLE;
static unsigned long last_wake_time = 0;

/* Command history buffer (static circular buffer) */
static CommandHistory cmd_history = {{}, 0, 0};

/* Last detection for duplicate filtering */
static DetectedKeyword last_detection = {LABEL_UNKNOWN, 0, 0.0f};

/* Posterior smoothing buffer (static circular buffer) */
static PosteriorFrame posterior_history[SMOOTHING_WINDOW];
static uint8_t posterior_head = 0;
static uint8_t posterior_count = 0;

/* Confidence accumulator */
static AccumulatedConfidence accumulated = {{0}};

/* ============================================================================
 * FUNCTION PROTOTYPES
 * ============================================================================ */

static void pdm_data_ready_callback(void);
static KeywordLabel string_to_label(const char* str);
static const char* label_to_string(KeywordLabel label);
static float calculate_energy(const int16_t* data, size_t length);
static float calculate_zcr(const int16_t* data, size_t length);
static bool check_vad_enhanced(const int16_t* data, size_t length);
static void get_smoothed_posteriors(ei_impulse_result_t* result, float* smoothed);
static void update_accumulator(const float* smoothed);
static int get_best_accumulated_label(float* out_score);
static void reset_accumulator(void);
static void reset_posterior_history(void);
static bool is_duplicate_detection(KeywordLabel label, float confidence, unsigned long current_time);
static void update_last_detection(KeywordLabel label, float confidence, unsigned long current_time);
static void history_push(DetectedKeyword item);
static void history_clear(void);
static void process_fsm(KeywordLabel label, float confidence, unsigned long timestamp);
static void execute_command(KeywordLabel device, KeywordLabel action);
static void RGB_control(bool red, bool green, bool blue);
static void fan_control(bool on);
static void beep_feedback(int duration_ms);

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

/* Convert label string from classifier to enum for efficient comparison */
static KeywordLabel string_to_label(const char* str) {
    if (strcmp(str, "WAKE") == 0) return LABEL_WAKE;
    if (strcmp(str, "ON") == 0) return LABEL_ON;
    if (strcmp(str, "OFF") == 0) return LABEL_OFF;
    if (strcmp(str, "LED") == 0) return LABEL_LED;
    if (strcmp(str, "FAN") == 0) return LABEL_FAN;
    if (strcmp(str, "_noise") == 0 || strcmp(str, "noise") == 0) return LABEL_NOISE;
    return LABEL_UNKNOWN;
}

/* Convert enum back to string for debugging output */
static const char* label_to_string(KeywordLabel label) {
    if (label >= 0 && label < LABEL_COUNT) {
        return LABEL_STRINGS[label];
    }
    return "_unknown";
}

/* ============================================================================
 * VOICE ACTIVITY DETECTION (VAD) FUNCTIONS
 * Enhanced VAD using both Energy and Zero-Crossing Rate
 * ============================================================================ */

/* Calculate average energy (mean squared amplitude) of audio samples */
static float calculate_energy(const int16_t* data, size_t length) {
    float energy = 0.0f;
    for (size_t i = 0; i < length; i++) {
        float sample = (float)data[i];
        energy += sample * sample;
    }
    return energy / (float)length;
}

/* Calculate Zero-Crossing Rate - ratio of sign changes in the signal
 * Lower ZCR typically indicates voiced speech, higher ZCR indicates noise/unvoiced */
static float calculate_zcr(const int16_t* data, size_t length) {
    if (length < 2) return 0.0f;

    int zero_crossings = 0;
    for (size_t i = 1; i < length; i++) {
        /* Count sign changes between consecutive samples */
        if ((data[i] >= 0 && data[i - 1] < 0) || (data[i] < 0 && data[i - 1] >= 0)) {
            zero_crossings++;
        }
    }
    return (float)zero_crossings / (float)(length - 1);
}

/* Enhanced VAD combining Energy and ZCR with hysteresis for stability */
static bool check_vad_enhanced(const int16_t* data, size_t length) {
    float energy = calculate_energy(data, length);
    float zcr = calculate_zcr(data, length);

    /* Normalize energy using log scale for better dynamic range */
    float energy_norm = (energy > 1.0f) ? (log10f(energy) / 10.0f) : 0.0f;
    if (energy_norm > 1.0f) energy_norm = 1.0f;

    /* Convert ZCR to speech likelihood score (lower ZCR = more likely speech) */
    float zcr_score = 1.0f - (zcr / VAD_ZCR_NOISE_MIN);
    if (zcr_score < 0.0f) zcr_score = 0.0f;
    if (zcr_score > 1.0f) zcr_score = 1.0f;

    /* Combine energy and ZCR scores with weighted average */
    float vad_score = VAD_ENERGY_WEIGHT * energy_norm + VAD_ZCR_WEIGHT * zcr_score;

    /* Apply hysteresis to prevent rapid state switching */
    float threshold = voice_detected ? (adaptive_vad_threshold * VAD_HYSTERESIS_FACTOR) : adaptive_vad_threshold;

    /* State machine for robust voice onset/offset detection */
    if (vad_score > threshold) {
        vad_active_frames++;
        vad_silence_frames = 0;
        if (vad_active_frames >= VAD_FRAMES_REQUIRED) {
            voice_detected = true;
        }
    } else {
        vad_silence_frames++;
        if (vad_silence_frames >= VAD_SILENCE_FRAMES_TIMEOUT) {
            vad_active_frames = 0;
            voice_detected = false;
        }
    }

    return voice_detected;
}

/* ============================================================================
 * POSTERIOR SMOOTHING FUNCTIONS
 * Moving average over multiple inference frames to reduce noise
 * ============================================================================ */

/* Add new posterior to history and compute smoothed output */
static void get_smoothed_posteriors(ei_impulse_result_t* result, float* smoothed) {
    /* Store current posteriors in circular buffer */
    PosteriorFrame* frame = &posterior_history[posterior_head];
    for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        frame->values[i] = result->classification[i].value;
    }
    frame->timestamp = millis();

    /* Advance circular buffer pointer */
    posterior_head = (posterior_head + 1) % SMOOTHING_WINDOW;
    if (posterior_count < SMOOTHING_WINDOW) {
        posterior_count++;
    }

    /* Compute moving average for each class */
    for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        float sum = 0.0f;
        for (uint8_t j = 0; j < posterior_count; j++) {
            sum += posterior_history[j].values[i];
        }
        smoothed[i] = sum / (float)posterior_count;
    }
}

/* Reset posterior smoothing history (call after successful detection) */
static void reset_posterior_history(void) {
    posterior_head = 0;
    posterior_count = 0;
    for (int i = 0; i < SMOOTHING_WINDOW; i++) {
        for (size_t j = 0; j < EI_CLASSIFIER_LABEL_COUNT; j++) {
            posterior_history[i].values[j] = 0.0f;
        }
    }
}

/* ============================================================================
 * CONFIDENCE ACCUMULATION FUNCTIONS
 * Require multiple frames of consistent detection before accepting
 * ============================================================================ */

/* Update accumulator with new smoothed posteriors using exponential decay */
static void update_accumulator(const float* smoothed) {
    for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        /* Apply decay to old value and add new contribution */
        accumulated.scores[i] = accumulated.scores[i] * ACCUMULATION_DECAY + smoothed[i];
    }
}

/* Find label with highest accumulated score */
static int get_best_accumulated_label(float* out_score) {
    int best_idx = -1;
    float best_val = 0.0f;

    for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (accumulated.scores[i] > best_val) {
            best_val = accumulated.scores[i];
            best_idx = (int)i;
        }
    }

    *out_score = best_val;
    return best_idx;
}

/* Reset accumulator (call after successful detection or timeout) */
static void reset_accumulator(void) {
    for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        accumulated.scores[i] = 0.0f;
    }
}

/* ============================================================================
 * DUPLICATE DETECTION FUNCTIONS
 * Prevent the same keyword from being detected multiple times in quick succession
 * ============================================================================ */

/* Check if detection is a duplicate of the previous one */
static bool is_duplicate_detection(KeywordLabel label, float confidence, unsigned long current_time) {
    /* Same label within cooldown period */
    if (label == last_detection.label && (current_time - last_detection.timestamp) < SAME_LABEL_COOLDOWN_MS) {
        /* Only accept if confidence is significantly higher */
        if (confidence <= last_detection.confidence + MIN_CONFIDENCE_DIFF) {
            return true; /* Duplicate - should be ignored */
        }
    }
    return false; /* Not a duplicate - should be processed */
}

/* Update last detection record */
static void update_last_detection(KeywordLabel label, float confidence, unsigned long current_time) {
    last_detection.label = label;
    last_detection.confidence = confidence;
    last_detection.timestamp = current_time;
}

/* ============================================================================
 * COMMAND HISTORY FUNCTIONS (Static circular buffer implementation)
 * ============================================================================ */

/* Push new keyword to history buffer */
static void history_push(DetectedKeyword item) {
    uint8_t idx;
    if (cmd_history.count < COMMAND_HISTORY_SIZE) {
        /* Buffer not full - add to end */
        idx = (cmd_history.head + cmd_history.count) % COMMAND_HISTORY_SIZE;
        cmd_history.count++;
    } else {
        /* Buffer full - overwrite oldest and advance head */
        idx = cmd_history.head;
        cmd_history.head = (cmd_history.head + 1) % COMMAND_HISTORY_SIZE;
    }
    cmd_history.items[idx] = item;
}

/* Clear all items from history */
static void history_clear(void) {
    cmd_history.head = 0;
    cmd_history.count = 0;
}

/* ============================================================================
 * FINITE STATE MACHINE (FSM) FOR COMMAND PROCESSING
 * ============================================================================ */

static void process_fsm(KeywordLabel label, float confidence, unsigned long timestamp) {
    unsigned long current_time = millis();

    /* Check for timeout in listening state */
    if (current_state == STATE_LISTENING) {
        if (current_time - last_wake_time > LISTENING_TIMEOUT_MS) {
            /* Timeout - return to idle state */
            current_state = STATE_IDLE;
            history_clear();
            reset_accumulator();
            reset_posterior_history();
            RGB_control(true, false, false); /* Red LED: sleeping */
            ei_printf("--- TIMEOUT: Returning to sleep ---\n");
            return;
        }
    }

    /* Skip if no valid label */
    if (label == LABEL_UNKNOWN || label == LABEL_NOISE) {
        return;
    }

    /* Process based on current state */
    switch (current_state) {
        case STATE_IDLE:
            /* Only respond to wake word */
            if (label == LABEL_WAKE) {
                current_state = STATE_LISTENING;
                last_wake_time = current_time;
                history_clear();
                RGB_control(false, false, true); /* Blue LED: listening */
                beep_feedback(150);              /* Short beep for feedback */
                ei_printf(">>> WAKE WORD DETECTED -> Listening... <<<\n");
            }
            break;

        case STATE_LISTENING:
            /* Update timeout timer */
            last_wake_time = current_time;

            /* Check for minimum interval between words */
            if (cmd_history.count > 0) {
                uint8_t last_idx = (cmd_history.head + cmd_history.count - 1) % COMMAND_HISTORY_SIZE;
                DetectedKeyword* last_item = &cmd_history.items[last_idx];

                /* Skip if same word detected too quickly */
                if (current_time - last_item->timestamp < MIN_WORDS_INTERVAL_MS) {
                    if (last_item->label == label) {
                        break; /* Skip duplicate */
                    }
                }
            }

            /* Add to history */
            DetectedKeyword new_item = {label, timestamp, confidence};
            history_push(new_item);

            /* Parse command from history - look for device + action pair */
            KeywordLabel action = LABEL_UNKNOWN;
            KeywordLabel device = LABEL_UNKNOWN;
            float action_score = 0.0f;
            float device_score = 0.0f;

            /* Iterate through history from newest to oldest */
            for (int i = cmd_history.count - 1; i >= 0; i--) {
                uint8_t idx = (cmd_history.head + i) % COMMAND_HISTORY_SIZE;
                DetectedKeyword* item = &cmd_history.items[idx];

                /* Skip entries outside command window */
                if (current_time - item->timestamp > COMMAND_WINDOW_MS) {
                    continue;
                }

                /* Calculate time-weighted score (newer = higher weight) */
                float time_weight = 1.0f - (float)(current_time - item->timestamp) / (float)COMMAND_WINDOW_MS;
                float weighted_score = item->confidence * time_weight;

                /* Check for action keywords (ON/OFF) */
                if (item->label == LABEL_ON || item->label == LABEL_OFF) {
                    if (weighted_score > action_score) {
                        action = item->label;
                        action_score = weighted_score;
                    }
                }

                /* Check for device keywords (LED/FAN) */
                if (item->label == LABEL_LED || item->label == LABEL_FAN) {
                    if (weighted_score > device_score) {
                        device = item->label;
                        device_score = weighted_score;
                    }
                }
            }

            /* Execute command if both device and action are found */
            if (action != LABEL_UNKNOWN && device != LABEL_UNKNOWN) {
                ei_printf(">>> COMMAND: %s %s (scores: %.2f, %.2f) <<<\n", label_to_string(device),
                          label_to_string(action), device_score, action_score);

                execute_command(device, action);

                /* Reset for next command */
                history_clear();
                last_detection = (DetectedKeyword){LABEL_UNKNOWN, 0, 0.0f};

                /* Visual feedback */
                RGB_control(false, true, false); /* Green LED: success */
                beep_feedback(100);
                delay(200);
                RGB_control(false, false, true); /* Return to blue */
            }
            break;
    }
}

/* Execute hardware command */
static void execute_command(KeywordLabel device, KeywordLabel action) {
    bool turn_on = (action == LABEL_ON);

    if (device == LABEL_LED) {
        digitalWrite(LED_BUILTIN, turn_on ? HIGH : LOW);
        ei_printf("LED turned %s\n", turn_on ? "ON" : "OFF");
    } else if (device == LABEL_FAN) {
        fan_control(turn_on);
        ei_printf("FAN turned %s\n", turn_on ? "ON" : "OFF");
    }
}

/* ============================================================================
 * HARDWARE CONTROL FUNCTIONS
 * ============================================================================ */

/* Control RGB LED - Note: LEDs are active-low on Nano 33 BLE */
static void RGB_control(bool red, bool green, bool blue) {
    digitalWrite(LEDR, red ? LOW : HIGH);
    digitalWrite(LEDG, green ? LOW : HIGH);
    digitalWrite(LEDB, blue ? LOW : HIGH);
}

/* Control fan motor */
static void fan_control(bool on) { digitalWrite(FAN_PIN, on ? HIGH : LOW); }

/* Generate beep feedback */
static void beep_feedback(int duration_ms) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(duration_ms);
    digitalWrite(BUZZER_PIN, LOW);
}

/* ============================================================================
 * PDM MICROPHONE CALLBACK (ISR)
 * ============================================================================ */

static void pdm_data_ready_callback(void) {
    int bytes_available = PDM.available();
    int bytes_read = PDM.read((char*)pdm_sample_buffer, bytes_available);
    int samples_read = bytes_read / 2; /* Each sample is 16-bit (2 bytes) */

    /* Transfer samples to ring buffer */
    for (int i = 0; i < samples_read; i++) {
        ring_buffer[write_index] = pdm_sample_buffer[i];
        write_index++;

        if (write_index >= RING_BUFFER_SIZE) {
            write_index = 0;
            buffer_filled_once = true;
        }
    }

    samples_since_last_inference += samples_read;
}

/* ============================================================================
 * SETUP - System initialization
 * ============================================================================ */

void setup() {
    /* Initialize serial for debugging */
    Serial.begin(115200);
    while (!Serial && millis() < 3000); /* Wait max 3 seconds for serial */

    Serial.println("\n========================================");
    Serial.println("Voice Control System - Nano 33 BLE Sense");
    Serial.println("========================================");

    /* Configure GPIO pins */
    pinMode(LED_BUILTIN, OUTPUT);
    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    pinMode(BUZZER_PIN, OUTPUT);
    pinMode(FAN_PIN, OUTPUT);

    /* Set initial state - Red LED indicates idle/sleeping */
    RGB_control(true, false, false);
    digitalWrite(BUZZER_PIN, LOW);
    digitalWrite(FAN_PIN, LOW);
    digitalWrite(LED_BUILTIN, LOW);

    /* Display classifier configuration */
    ei_printf("Classifier settings:\n");
    ei_printf("  Sample interval: %.2f ms\n", (float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf("  Frame size: %d samples\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    ei_printf("  Label count: %d\n", EI_CLASSIFIER_LABEL_COUNT);
    ei_printf("  Inference interval: %d ms\n", INFERENCE_EVERY_MS);

    /* Configure PDM microphone */
    PDM.onReceive(pdm_data_ready_callback);
    PDM.setBufferSize(PDM_BUFFER_SIZE);
    PDM.setGain(127); /* Maximum gain for better sensitivity */

    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        ei_printf("ERROR: Failed to start PDM microphone!\n");
        while (1) {
            RGB_control(true, false, false);
            delay(200);
            RGB_control(false, false, false);
            delay(200);
        }
    }

    ei_printf("System ready. Say 'WAKE' to activate.\n\n");

    /* Startup beep */
    beep_feedback(100);
}

/* ============================================================================
 * MAIN LOOP - Continuous inference and command processing
 * ============================================================================ */

void loop() {
    /* Calculate samples needed for desired inference interval */
    int samples_to_wait = (INFERENCE_EVERY_MS * EI_CLASSIFIER_FREQUENCY) / 1000;

    /* Check if enough samples collected */
    if (samples_since_last_inference < samples_to_wait) {
        return;
    }
    samples_since_last_inference = 0;

    /* Wait for buffer to fill initially */
    if (!buffer_filled_once && write_index < EI_CLASSIFIER_RAW_SAMPLE_COUNT) {
        return;
    }

    /* ---- Voice Activity Detection ---- */
    /* Extract recent samples for VAD analysis */
    int16_t vad_samples[VAD_SAMPLE_SIZE];
    int start_idx = (write_index - VAD_SAMPLE_SIZE + RING_BUFFER_SIZE) % RING_BUFFER_SIZE;
    for (int i = 0; i < VAD_SAMPLE_SIZE; i++) {
        vad_samples[i] = ring_buffer[(start_idx + i) % RING_BUFFER_SIZE];
    }

    bool has_voice = check_vad_enhanced(vad_samples, VAD_SAMPLE_SIZE);

    /* Skip inference if no voice and in idle state (save power) */
    if (!has_voice && current_state == STATE_IDLE) {
        /* Periodically reset accumulator during silence */
        reset_accumulator();
        return;
    }

    /* ---- Prepare signal for classifier ---- */
    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data = [](size_t offset, size_t length, float* out_ptr) -> int {
        int read_start = write_index - EI_CLASSIFIER_RAW_SAMPLE_COUNT + offset;
        if (read_start < 0) read_start += RING_BUFFER_SIZE;

        for (size_t i = 0; i < length; i++) {
            int idx = (read_start + i) % RING_BUFFER_SIZE;
            out_ptr[i] = (float)ring_buffer[idx];
        }
        return 0;
    };

    /* ---- Run classifier ---- */
    ei_impulse_result_t result = {0};
    EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
    if (err != EI_IMPULSE_OK) {
        ei_printf("Classifier error: %d\n", err);
        return;
    }

    /* ---- Posterior smoothing ---- */
    float smoothed[EI_CLASSIFIER_LABEL_COUNT];
    get_smoothed_posteriors(&result, smoothed);

    /* ---- Find best class from smoothed posteriors ---- */
    int best_class_idx = -1;
    float best_smoothed = 0.0f;
    for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (smoothed[i] > best_smoothed) {
            best_smoothed = smoothed[i];
            best_class_idx = (int)i;
        }
    }

    /* Get label and check if it's noise/unknown */
    const char* label_str = result.classification[best_class_idx].label;
    KeywordLabel label = string_to_label(label_str);

    /* ---- FILTER NOISE BEFORE ACCUMULATION ---- */
    if (label == LABEL_UNKNOWN || label == LABEL_NOISE) {
        /* Decay accumulator during noise/silence instead of accumulating */
        for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
            accumulated.scores[i] *= 0.5f; /* Fast decay during noise */
        }
        return; /* Skip further processing */
    }

    /* ---- Confidence accumulation ---- */
    update_accumulator(smoothed);

    float accumulated_score = 0.0f;
    best_class_idx = get_best_accumulated_label(&accumulated_score);

    /* Check if accumulated score exceeds threshold */
    if (best_class_idx < 0 || accumulated_score < ACCUMULATION_THRESHOLD) {
        /* Not enough confidence yet - check for timeout in listening state */
        if (current_state == STATE_LISTENING) {
            unsigned long current_time = millis();
            if (current_time - last_wake_time > LISTENING_TIMEOUT_MS) {
                current_state = STATE_IDLE;
                history_clear();
                reset_accumulator();
                reset_posterior_history();
                RGB_control(true, false, false);
                ei_printf("--- TIMEOUT: Returning to sleep ---\n");
            }
        }
        return;
    }

    /* Get label info */
    label_str = result.classification[best_class_idx].label;
    label = string_to_label(label_str);
    float confidence = smoothed[best_class_idx];
    unsigned long current_time = millis();

    /* Debug output */
    ei_printf("[%.2f] %s (acc=%.2f, smooth=%.2f)\n", (float)current_time / 1000.0f, label_str, accumulated_score,
              confidence);

    /* Filter out noise and unknown */
    if (label == LABEL_UNKNOWN || label == LABEL_NOISE) {
        return;
    }

    /* Check confidence threshold */
    if (confidence < PREDICTION_THRESHOLD) {
        return;
    }

    /* Check for duplicate */
    if (is_duplicate_detection(label, confidence, current_time)) {
        ei_printf("  -> Skipped (duplicate)\n");
        return;
    }

    /* Valid detection - update state and process */
    update_last_detection(label, confidence, current_time);
    reset_accumulator();
    reset_posterior_history();

    ei_printf("  -> DETECTED: %s\n", label_str);

    /* Process through FSM */
    process_fsm(label, confidence, current_time);
}
