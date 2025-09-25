// ESP32 Arduino (core v3+) Sketch: Photic Entrainment Presets + Idle Pattern
// LEDs: GPIO 2 (IDLE) and GPIO 13 (ACTIVE modes) using new LEDC API
// Button: GPIO 0 (BOOT) - active LOW with internal pull-up

#include <Arduino.h>

// ---------- Pins ----------
constexpr int LED_PIN_IDLE  = 2;   // used ONLY in IDLE
constexpr int LED_PIN_ACTIVE= 13;  // used ONLY in ACTIVE modes
constexpr int LED_PIN_ACTIVE2= 12;  // used ONLY in ACTIVE modes
constexpr int BUTTON_PIN    = 0;

// ---------- LEDC (hardware PWM) ----------
constexpr uint8_t  LEDC_TIMER_BIT   = 8;      // 8-bit resolution (0..255)
constexpr uint32_t LEDC_CARRIER_HZ  = 1000;   // PWM carrier; we modulate duty at flicker freq

// ---------- Idle pattern ----------
constexpr uint32_t IDLE_ON_MS   = 200;
constexpr uint32_t IDLE_GAP_MS  = 200;
constexpr uint32_t IDLE_LONG_MS = 2000;

// ---------- Waveforms ----------
enum Waveform : uint8_t { WAVE_SQUARE = 0, WAVE_SMOOTH = 1 };

// ---------- Mode preset ----------
struct ModePreset {
  const char* name;
  const char* description;
  float  freqHz;          // flicker frequency
  float  dutyFrac;        // 0.0..1.0
  Waveform waveform;      // SQUARE or SMOOTH (raised-cosine)
  uint8_t brightness;     // 0..255 peak
  uint32_t sessionSec;    // duration in seconds
  uint32_t rampSec;       // ramp-in seconds
  uint8_t ledSel; // which LED
};

// Suggested safe starter presets
ModePreset PRESETS[] = {
  { "Relax / Alpha (10 Hz)",   "Gentle relaxation; alpha-band smoothed flicker.", 10.0f, 0.40f, WAVE_SMOOTH, 15, 10*60, 5,1 },
  { "Meditation / Theta (6 Hz)","Calming, drowsy; theta-band gentle flicker.",     6.0f, 0.30f, WAVE_SMOOTH, 10, 10*60, 6,1 },
  { "Alert / Beta (16 Hz)",    "Mild alertness; gentle edges, low brightness.",    16.0f, 0.30f, WAVE_SMOOTH,  8,  4*60, 5,1 },
  { "Cognitive / Gamma (40 Hz)","Experimental gamma; isochronic-like pulses.",     40.0f, 0.30f, WAVE_SQUARE, 14, 10*60, 5,1 },
  { "Relax / Alpha (10 Hz)",   "Gentle relaxation; alpha-band smoothed flicker.", 10.0f, 0.40f, WAVE_SMOOTH, 150, 10*60, 5,2 },
  { "Meditation / Theta (6 Hz)","Calming, drowsy; theta-band gentle flicker.",     6.0f, 0.30f, WAVE_SMOOTH, 100, 10*60, 6,2 },
  { "Alert / Beta (16 Hz)",    "Mild alertness; gentle edges, low brightness.",    16.0f, 0.30f, WAVE_SMOOTH,  80,  4*60, 5,2 },
  { "Cognitive / Gamma (40 Hz)","Experimental gamma; isochronic-like pulses.",     40.0f, 0.30f, WAVE_SQUARE, 140, 10*60, 5,2 },
  { "Relax / Alpha (10 Hz)",   "Gentle relaxation; alpha-band smoothed flicker.", 10.0f, 0.40f, WAVE_SMOOTH, 250, 10*60, 5,2 },
  { "Meditation / Theta (6 Hz)","Calming, drowsy; theta-band gentle flicker.",     6.0f, 0.30f, WAVE_SMOOTH, 200, 10*60, 6,2 },
  { "Alert / Beta (16 Hz)",    "Mild alertness; gentle edges, low brightness.",    16.0f, 0.30f, WAVE_SMOOTH,  150,  4*60, 5,2 },
  { "Cognitive / Gamma (40 Hz)","Experimental gamma; isochronic-like pulses.",     40.0f, 0.30f, WAVE_SQUARE, 240, 10*60, 5,2 }
};
constexpr int NUM_PRESETS = sizeof(PRESETS)/sizeof(PRESETS[0]);

// ---------- State machine ----------
enum RunState : uint8_t { STATE_IDLE = 0, STATE_ACTIVE = 1 };
RunState runState = STATE_IDLE;
int currentPreset = -1;

uint32_t stateStartMs = 0;  // when current state started
uint32_t modeStartMs  = 0;  // when active preset started

// Button debounce
bool lastBtn = true;             // pull-up -> idle HIGH
uint32_t lastDebounceMs = 0;
constexpr uint32_t DEBOUNCE_MS = 30;

// ---------- Helpers ----------
inline void writeIdle(uint8_t duty)   { ledcWrite(LED_PIN_IDLE, duty); }
inline void writeActive(uint8_t duty) { ledcWrite(LED_PIN_ACTIVE, duty); ledcWrite(LED_PIN_ACTIVE2, duty); }
inline void writeActiveGreen(uint8_t duty) { ledcWrite(LED_PIN_ACTIVE2, duty);  }
inline void writeActiveRed(uint8_t duty) {  ledcWrite(LED_PIN_ACTIVE, duty); }

uint8_t smoothPulseDuty(float phase, float dutyFrac, uint8_t peak) {
  if (dutyFrac <= 0.0f) return 0;
  while (phase >= 1.0f) phase -= 1.0f;
  if (phase < 0.0f) phase = 0.0f;
  if (phase >= dutyFrac) return 0;
  float x = phase / dutyFrac; // 0..1 over the pulse
  float env = 0.5f * (1.0f - cosf(2.0f * PI * x)); // 0..1 raised-cosine
  int val = int(env * peak + 0.5f);
  if (val < 0) val = 0; if (val > 255) val = 255;
  return (uint8_t)val;
}

inline uint8_t squarePulseDuty(float phase, float dutyFrac, uint8_t peak) {
  while (phase >= 1.0f) phase -= 1.0f;
  return (phase < dutyFrac) ? peak : 0;
}

// Brightness ramp (0..1)
float rampScale(uint32_t elapsedMs, uint32_t rampSec) {
  if (rampSec == 0) return 1.0f;
  float t = elapsedMs / 1000.0f;
  if (t >= (float)rampSec) return 1.0f;
  return t / (float)rampSec;
}

void printPresetInfo(const ModePreset& p, int index) {
  Serial.println();
  Serial.println(F("=== ENTERING MODE ==="));
  Serial.print(F("Index: ")); Serial.println(index);
  Serial.print(F("Name: ")); Serial.println(p.name);
  Serial.print(F("Description: ")); Serial.println(p.description);
  Serial.print(F("Frequency (Hz): ")); Serial.println(p.freqHz, 3);
  Serial.print(F("Duty (%): ")); Serial.println(p.dutyFrac * 100.0f, 1);
  Serial.print(F("Waveform: ")); Serial.println(p.waveform == WAVE_SQUARE ? "SQUARE" : "SMOOTH (raised-cosine)");
  Serial.print(F("Peak Brightness (0..255): ")); Serial.println(p.brightness);
  Serial.print(F("Session (sec): ")); Serial.println(p.sessionSec);
  Serial.print(F("Ramp-in (sec): ")); Serial.println(p.rampSec);
  Serial.print(F("LED Sel: ")); Serial.println(p.ledSel);
  Serial.println(F("====================="));
}

void printIdleInfo() {
  Serial.println();
  Serial.println(F("--- IDLE MODE ---"));
  Serial.println(F("LED on GPIO 2 ONLY."));
  Serial.println(F("Pattern: 200ms ON, 200ms OFF, 200ms ON, then 2000ms OFF, repeat."));
}

// Start a preset by index
void startPreset(int idx) {
  currentPreset = idx;
  runState = STATE_ACTIVE;
  stateStartMs = millis();
  modeStartMs  = stateStartMs;

  // Ensure LED routing: ACTIVE -> GPIO13 only, turn off GPIO2
  writeIdle(0);
  printPresetInfo(PRESETS[currentPreset], currentPreset);
}

// Return to idle
void startIdle() {
  runState = STATE_IDLE;
  currentPreset = -1;
  stateStartMs = millis();

  // Ensure LED routing: IDLE -> GPIO2 only, turn off GPIO13
  writeActive(0);
  printIdleInfo();
}

// Handle button with debounce; return true on *new* press
bool buttonPressed() {
  bool raw = digitalRead(BUTTON_PIN);     // HIGH = not pressed (pull-up), LOW = pressed
  uint32_t now = millis();
  if (raw != lastBtn) {
    lastDebounceMs = now;
    lastBtn = raw;
  }
  if ((now - lastDebounceMs) > DEBOUNCE_MS) {
    if (raw == LOW) {
      // wait for release to avoid auto-repeat
      while (digitalRead(BUTTON_PIN) == LOW) { delay(1); }
      delay(10);
      return true;
    }
  }
  return false;
}

// ---------- Setup ----------
void setup() {
  Serial.begin(115200);
  delay(100);

  // Button
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  // NEW LEDC API (core v3+): attach pins directly
  // ledcAttach(pin, freq_hz, resolution_bits)
  ledcAttach(LED_PIN_IDLE,   LEDC_CARRIER_HZ, LEDC_TIMER_BIT);
  ledcAttach(LED_PIN_ACTIVE, LEDC_CARRIER_HZ, LEDC_TIMER_BIT);
  ledcAttach(LED_PIN_ACTIVE2, LEDC_CARRIER_HZ, LEDC_TIMER_BIT);

  // Start in Idle
  startIdle();
}

// ---------- Loop ----------
void loop() {
  // Button logic:
  // - If IDLE and pressed -> start preset 0
  // - If ACTIVE and pressed:
  //      if NOT last preset -> next preset
  //      if last preset     -> go to IDLE   (requested behavior)
  if (buttonPressed()) {
    if (runState == STATE_IDLE) {
      startPreset(0);
    } else {
      if (currentPreset < NUM_PRESETS - 1) startPreset(currentPreset + 1);
      else {
        Serial.println(F("End of preset list -> returning to IDLE."));
        startIdle();
      }
    }
  }

  uint32_t nowMs = millis();

  if (runState == STATE_IDLE) {
    // Only GPIO2 should blink; ensure GPIO13 stays OFF
    writeActive(0);

    // Idle: ON 200ms, OFF 200ms, ON 200ms, OFF 2000ms
    uint32_t cycle = IDLE_ON_MS + IDLE_GAP_MS + IDLE_ON_MS + IDLE_LONG_MS; // 2600ms
    uint32_t t = (nowMs - stateStartMs) % cycle;

    uint8_t idleDuty = 60; // soft blink on LED 2
    if (t < IDLE_ON_MS) {
      writeIdle(idleDuty);
    } else if (t < IDLE_ON_MS + IDLE_GAP_MS) {
      writeIdle(0);
    } else if (t < IDLE_ON_MS + IDLE_GAP_MS + IDLE_ON_MS) {
      writeIdle(idleDuty);
    } else {
      writeIdle(0);
    }
    delay(1);
    return;
  }

  // ACTIVE mode: only GPIO13 should run; ensure GPIO2 stays OFF
  writeIdle(0);

  const ModePreset& p = PRESETS[currentPreset];

  // Session timeout -> back to idle
  uint32_t elapsedMs = nowMs - modeStartMs;
  if (elapsedMs >= p.sessionSec * 1000UL) {
    Serial.println(F("Session completed -> returning to IDLE."));
    startIdle();
    return;
  }

  // Compute phase in current period [0,1)
  uint32_t nowUs = micros();
  float periodUs = 1000000.0f / p.freqHz;
  float phase = fmodf((float)(nowUs), periodUs) / periodUs; // 0..1

  // Ramp-in brightness
  float ramp = rampScale(elapsedMs, p.rampSec);
  uint8_t peak = (uint8_t)(p.brightness * ramp + 0.5f);

  // Duty based on waveform
  uint8_t outDuty = (p.waveform == WAVE_SQUARE)
                      ? squarePulseDuty(phase, p.dutyFrac, peak)
                      : smoothPulseDuty(phase, p.dutyFrac, peak);

  // Drive ONLY the ACTIVE LED (GPIO13)
  writeActive(outDuty);
  // if (p.ledSel == 1) {
  //   writeActiveGreen(outDuty);
  //   writeActiveRed(0);
  // }
  // if (p.ledSel == 2) {
  //   writeActiveRed(outDuty);
  //   writeActiveGreen(0);
  // }
  

  // keep jitter low; loop fast
  delayMicroseconds(500);
}
