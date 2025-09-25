// File: esp32_8ch_scope_option1.ino
// Pure Arduino API (no ESP-IDF headers). 8 channels sampled via analogRead()
// in a tight loop. Streams framed binary over USB-Serial.
//
// Frame header (little-endian):
//   'O','S'                 (2 bytes magic)
//   uint8_t  version        (1)
//   uint8_t  channels       (8)
//   uint16_t n_sets         (# of per-channel samples in this frame)
//   uint32_t samp_rate_total_hz  (aggregate 8-ch sample rate)
//   uint32_t seq            (frame counter)
// Payload:
//   n_sets * channels * uint16_t (12-bit ADC value left-shifted to 16-bit)
//
// Interleaving: ch0,ch1,...,ch7, ch0,ch1,...,ch7, ...

#include <Arduino.h>

// ---------------- User settings ----------------
static const uint32_t SERIAL_BAUD              = 921600;   // Try 2,000,000. You can test 3,000,000 if your USB-UART supports it.
static const uint32_t PER_CHANNEL_SPS_TARGET   = 10000;     // per-channel samples/sec target (real will be approximate)
static const uint16_t FRAME_INTERLEAVED_SETS   = 256;       // samples per channel per frame (payload = 256*8*2 = 4096 bytes)
static const bool     ALIGN_12_TO_16_BITS_MSb  = true;      // left-shift 12b to MSBs (GUI expects this; safe to keep true)
// ------------------------------------------------

// ADC1-only pins (8 channels) to keep things simple & Wi-Fi friendly
// CH order matches the array order below.
static const int CH_PINS[8] = {
  36, // ADC1_CH0  (GPIO36)
  39, // ADC1_CH3  (GPIO39)
  34, // ADC1_CH6  (GPIO34)
  35, // ADC1_CH7  (GPIO35)
  32, // ADC1_CH4  (GPIO32)
  33, // ADC1_CH5  (GPIO33)
  37, // ADC1_CH1  (GPIO37)
  38  // ADC1_CH2  (GPIO38)
};

// Frame header (packed)
struct __attribute__((packed)) FrameHeader {
  char     magic[2];           // 'O','S'
  uint8_t  version;            // 1
  uint8_t  channels;           // 8
  uint16_t n_sets;             // samples per channel in this frame
  uint32_t samp_rate_total_hz; // aggregate sample rate across all channels
  uint32_t seq;                // frame counter
};

// Buffers
static uint16_t frame_buf[FRAME_INTERLEAVED_SETS * 8];
static uint32_t frame_seq = 0;

// Timing helper to *aim* for a per-channel rate (best-effort with analogRead)
static inline void busy_wait_until(uint32_t target_us) {
  while ((uint32_t)micros() < target_us) {
    // tight spin
  }
}

void setup() {
  // Serial first
  Serial.begin(SERIAL_BAUD);
  delay(200);

  // Configure ADC resolution & attenuation
  // ESP32 Arduino: analogSetWidth(9..12)
  analogSetWidth(12); // 12-bit (0..4095)

  // Set per-pin attenuation to ~0–3.3V range (11 dB)
  // If your core supports per-pin attenuation:
  for (int i = 0; i < 8; ++i) {
    analogSetPinAttenuation(CH_PINS[i], ADC_11db);
  }
  // Otherwise, globally:a
  // analogSetAttenuation(ADC_11db);

  // Optional: pre-attach pins (some cores provide adcAttachPin for a tiny speed win)
  // for (int i = 0; i < 8; ++i) adcAttachPin(CH_PINS[i]);

  // Tiny banner
  // Serial.println("ESP32 8CH analogRead() streaming");
}

// Measure analogRead() overhead once (rough estimate)
static uint32_t estimate_read_us() {
  uint32_t t0 = micros();
  volatile uint16_t sink = 0;
  for (int i = 0; i < 100; ++i) {
    sink += analogRead(CH_PINS[i & 7]);
  }
  uint32_t dt = micros() - t0;
  return dt / 100;
}

void loop() {
  static bool init_done = false;
  static uint32_t read_us_est = 120; // conservative default per analogRead call
  if (!init_done) {
    // Roughly estimate analogRead cost on *your* board
    read_us_est = estimate_read_us();  // often ~80–150 us depending on core/settings
    init_done = true;
  }

  // We’ll *aim* timing by spacing the start of each 8-channel set.
  // Target period per *channel*:
  const float per_channel_period_us_f = 1e6f / (float)PER_CHANNEL_SPS_TARGET;
  const uint32_t per_set_period_us = (uint32_t)(per_channel_period_us_f); // for each channel read we try to respect per-channel pacing

  // But we read channels back-to-back; the time to read 8 channels is ~8*read_us_est.
  // We'll pace each *channel* read; simple and "good enough" for option 1.

  uint32_t next_read_time_us = micros();

  // Fill one frame = FRAME_INTERLEAVED_SETS groups of 8 samples
  for (uint16_t set_idx = 0; set_idx < FRAME_INTERLEAVED_SETS; ++set_idx) {
    for (uint8_t ch = 0; ch < 8; ++ch) {
      // Pace this channel read (best effort)
      next_read_time_us += per_set_period_us;
      busy_wait_until(next_read_time_us);

      // Acquire sample
      uint16_t v = (uint16_t)analogRead(CH_PINS[ch]);  // 0..4095 nominal
      if (ALIGN_12_TO_16_BITS_MSb) v <<= 4;            // align (12->16)
      frame_buf[set_idx * 8 + ch] = v;
    }
  }

  // Compute an *approximate* achieved total sample rate.
  // Each set collected 8 channel-reads spaced by per_set_period_us (not perfect).
  // Use target unless you prefer to measure real time elapsed.
  const uint32_t samp_rate_total_hz = PER_CHANNEL_SPS_TARGET * 8;

  // Emit header
  FrameHeader H;
  H.magic[0] = 'O'; H.magic[1] = 'S';
  H.version = 1;
  H.channels = 8;
  H.n_sets = FRAME_INTERLEAVED_SETS;
  H.samp_rate_total_hz = samp_rate_total_hz;
  H.seq = frame_seq++;

  Serial.write((uint8_t*)&H, sizeof(H));
  Serial.write((uint8_t*)frame_buf, sizeof(frame_buf));

  // Optional tiny yield
  // delay(0);
}
