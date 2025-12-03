/**
 * API Client for leaderboard - uses server API for shared leaderboard
 */

// Reference network size for efficiency calculation (6→64→64→2)
const REFERENCE_PARAMS = 8706

// API base URL - use relative path in production (Caddy proxies /api to backend)
const API_BASE = import.meta.env.VITE_API_URL || '/api'

export interface LeaderboardEntry {
  id: string
  name: string
  score: number           // Efficiency-adjusted score
  pipes: number           // Raw pipes passed
  params: number          // Total network parameters
  architecture: string    // e.g. "6→64→64→2"
  createdAt: string
  isChampion?: boolean
  isYou?: boolean
}

/**
 * Calculate efficiency-adjusted score
 * Smaller networks get bonus points: score = pipes * sqrt(reference / actual_params)
 */
export function calculateAdjustedScore(pipes: number, params: number): number {
  if (params <= 0) return pipes
  const efficiency = Math.sqrt(REFERENCE_PARAMS / params)
  return Math.round(pipes * efficiency * 10) / 10  // Round to 1 decimal
}

/**
 * Get efficiency multiplier for display
 */
export function getEfficiencyMultiplier(params: number): number {
  if (params <= 0) return 1
  return Math.sqrt(REFERENCE_PARAMS / params)
}

export interface LeaderboardResponse {
  entries: LeaderboardEntry[]
  champion?: LeaderboardEntry
}

export interface SubmitScoreRequest {
  name: string
  pipes: number           // Raw pipes passed
  params: number          // Network parameters
  architecture: string
}

export interface SubmitScoreResponse {
  success: boolean
  entry: LeaderboardEntry
  isNewChampion: boolean
}

class ApiClient {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl
  }

  /**
   * Get leaderboard entries from server
   */
  async getLeaderboard(limit: number = 10): Promise<LeaderboardResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/leaderboard?limit=${limit}`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      return await response.json()
    } catch (error) {
      console.warn('Failed to fetch leaderboard from server:', error)
      // Return empty leaderboard on error
      return { entries: [], champion: undefined }
    }
  }

  /**
   * Submit a new score to the leaderboard (saves to server)
   */
  async submitScore(request: SubmitScoreRequest): Promise<SubmitScoreResponse> {
    // Calculate efficiency-adjusted score
    const adjustedScore = calculateAdjustedScore(request.pipes, request.params)

    try {
      const response = await fetch(`${this.baseUrl}/leaderboard`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: request.name,
          pipes: request.pipes,
          params: request.params,
          architecture: request.architecture,
          score: adjustedScore,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      // Mark the entry as "you" on the client side
      if (result.entry) {
        result.entry.isYou = true
      }
      return result
    } catch (error) {
      console.error('Failed to submit score:', error)
      // Return a failed response
      return {
        success: false,
        entry: {
          id: '',
          name: request.name,
          score: adjustedScore,
          pipes: request.pipes,
          params: request.params,
          architecture: request.architecture,
          createdAt: new Date().toISOString(),
        },
        isNewChampion: false,
      }
    }
  }

  /**
   * Get the lowest score on the leaderboard (or 0 if empty/less than 10 entries)
   */
  async getLowestScore(): Promise<number> {
    try {
      const response = await fetch(`${this.baseUrl}/leaderboard/lowest`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      return data.lowestScore || 0
    } catch (error) {
      console.warn('Failed to get lowest score:', error)
      return 0  // Allow submission on error
    }
  }
}

// Export singleton instance
export const apiClient = new ApiClient()

// Export class for custom instantiation
export { ApiClient }




