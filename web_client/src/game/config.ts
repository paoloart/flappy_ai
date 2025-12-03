/**
 * Game configuration constants
 * Ported from Python FlapPyBird for accurate physics
 */

export const GameConfig = {
  // Canvas dimensions (original Flappy Bird size - scaled via CSS)
  WIDTH: 288,
  HEIGHT: 512,
  VIEWPORT_HEIGHT: 400, // Height above floor

  // Bird physics
  BIRD: {
    X: 57, // Fixed x position (20% of width)
    INITIAL_Y: 244, // Middle of screen
    WIDTH: 34,
    HEIGHT: 24,
    FLAP_VELOCITY: -9, // Upward velocity on flap
    GRAVITY: 1, // Acceleration per frame
    MAX_VELOCITY_DOWN: 10,
    MAX_VELOCITY_UP: -8,
    ROTATION_SPEED: -3,
    ROTATION_MIN: -90,
    ROTATION_MAX: 20,
    MIN_Y: -48, // Allow going slightly above screen
    // Animation
    FRAME_RATE: 5, // Frames per wing flap cycle
  },

  // Pipe configuration
  PIPE: {
    WIDTH: 52,
    HEIGHT: 320,
    GAP: 120, // Gap between upper and lower pipes
    VELOCITY: -5, // Horizontal speed
    SPAWN_DISTANCE: 182, // Distance between pipe centers (width * 3.5)
    INITIAL_X_OFFSET: 468, // First pipe x position (width + width*3)
  },

  // Floor
  FLOOR: {
    HEIGHT: 112,
    VELOCITY: -5, // Same as pipe velocity for parallax
  },

  // Scoring
  SCORE: {
    PASS_PIPE: 1,
    STEP_PENALTY: -0.01,
    DEATH_PENALTY: -1,
  },

  // Game loop
  FPS: 30,
  FRAME_TIME: 1000 / 30,
} as const

export type GameAction = 0 | 1 // 0 = idle, 1 = flap

/**
 * Reward configuration - can be toggled/scaled before training starts
 */
export interface RewardConfig {
  stepPenalty: number
  passPipe: number
  deathPenalty: number
  flapCost: number
  centerReward: number
  // Note: outOfBoundsCost is hardcoded internally (0.1 penalty for y < 0)
}

export const DefaultRewardConfig: RewardConfig = {
  stepPenalty: -0.01,
  passPipe: 1.0,
  deathPenalty: -1.0,
  flapCost: 0.003,         // Match Python default
  centerReward: 0.01,      // Shaping: small reward for moving toward gap center
}

/**
 * Observation feature flags - can be toggled before training starts
 */
export interface ObservationConfig {
  birdY: boolean
  birdVel: boolean
  dx1: boolean
  dy1: boolean
  dx2: boolean
  dy2: boolean
  gapVel1: boolean
  gapVel2: boolean
}

export const DefaultObservationConfig: ObservationConfig = {
  birdY: true,
  birdVel: true,
  dx1: true,
  dy1: true,
  dx2: true,
  dy2: true,
  gapVel1: false,
  gapVel2: false,
}

/**
 * Network architecture config - set before training starts
 */
export interface NetworkConfig {
  hiddenLayers: number[]
  inputFeatures: ObservationConfig
}

export const DefaultNetworkConfig: NetworkConfig = {
  hiddenLayers: [128, 128],
  inputFeatures: DefaultObservationConfig,
}




