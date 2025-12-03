/**
 * API Client for leaderboard - uses local storage and JSON file
 */

// Reference network size for efficiency calculation (6→64→64→2)
const REFERENCE_PARAMS = 8706

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

// Local storage key for leaderboard data
const LEADERBOARD_STORAGE_KEY = 'flappy-ai-leaderboard'

class ApiClient {
  /**
   * Get leaderboard entries from local storage
   */
  async getLeaderboard(limit: number = 10): Promise<LeaderboardResponse> {
    try {
      const storedData = localStorage.getItem(LEADERBOARD_STORAGE_KEY)
      if (storedData) {
        const data = JSON.parse(storedData) as LeaderboardResponse
        return this.processLeaderboard(data, limit)
      }
    } catch (error) {
      console.warn('Failed to load leaderboard:', error)
    }

    // Return empty leaderboard if nothing stored yet
    return { entries: [], champion: undefined }
  }

  /**
   * Process leaderboard: sort by score, limit entries, mark champion
   */
  private processLeaderboard(data: LeaderboardResponse, limit: number): LeaderboardResponse {
    // Sort by score descending
    const sortedEntries = [...data.entries].sort((a, b) => b.score - a.score)
    
    // Limit entries
    const limitedEntries = sortedEntries.slice(0, limit)
    
    // Mark champion
    limitedEntries.forEach((entry, index) => {
      entry.isChampion = index === 0
    })

    const champion = limitedEntries.length > 0 ? limitedEntries[0] : undefined

    return {
      entries: limitedEntries,
      champion,
    }
  }

  /**
   * Submit a new score to the leaderboard (saves to localStorage)
   */
  async submitScore(request: SubmitScoreRequest): Promise<SubmitScoreResponse> {
    // Load current leaderboard
    const current = await this.getLeaderboard(100) // Get all entries
    
    // Calculate efficiency-adjusted score
    const adjustedScore = calculateAdjustedScore(request.pipes, request.params)
    
    // Create new entry
    const newEntry: LeaderboardEntry = {
      id: `entry-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      name: request.name,
      score: adjustedScore,
      pipes: request.pipes,
      params: request.params,
      architecture: request.architecture,
      createdAt: new Date().toISOString(),
      isYou: true,
    }

    // Check if this is a new champion
    const currentChampion = current.champion
    const isNewChampion = !currentChampion || adjustedScore > currentChampion.score

    // Add new entry to the list
    const updatedEntries = [...current.entries, newEntry]
    
    // Sort by adjusted score
    updatedEntries.sort((a, b) => b.score - a.score)

    // Save back to localStorage
    const updatedData: LeaderboardResponse = {
      entries: updatedEntries,
      champion: isNewChampion ? newEntry : currentChampion,
    }
    localStorage.setItem(LEADERBOARD_STORAGE_KEY, JSON.stringify(updatedData))

    return {
      success: true,
      entry: newEntry,
      isNewChampion,
    }
  }

  /**
   * Get the lowest score on the leaderboard (or 0 if empty/less than 10 entries)
   */
  async getLowestScore(): Promise<number> {
    const data = await this.getLeaderboard(10)
    if (data.entries.length < 10) return 0  // Always qualify if less than 10 entries
    return data.entries[data.entries.length - 1]?.score || 0
  }

  /**
   * Clear the leaderboard (for testing)
   */
  clearLeaderboard(): void {
    localStorage.removeItem(LEADERBOARD_STORAGE_KEY)
  }
}

// Export singleton instance
export const apiClient = new ApiClient()

// Export class for custom instantiation
export { ApiClient }




