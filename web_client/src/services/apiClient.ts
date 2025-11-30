/**
 * API Client for leaderboard and champion model endpoints
 */

export interface LeaderboardEntry {
  id: string
  name: string
  score: number
  createdAt: string
  isChampion?: boolean
  isYou?: boolean
}

export interface LeaderboardResponse {
  entries: LeaderboardEntry[]
  champion?: LeaderboardEntry
}

export interface SubmitScoreRequest {
  name: string
  score: number
  modelWeights?: ArrayBuffer
}

export interface SubmitScoreResponse {
  success: boolean
  entry: LeaderboardEntry
  isNewChampion: boolean
}

export interface ChampionModel {
  name: string
  score: number
  weightsUrl: string
}

// Base API URL - can be configured via environment variable
const API_BASE = import.meta.env.VITE_API_URL || '/api'

class ApiClient {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl
  }

  /**
   * Get leaderboard entries
   */
  async getLeaderboard(limit: number = 10): Promise<LeaderboardResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/leaderboard?limit=${limit}`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      return await response.json()
    } catch (error) {
      console.warn('Failed to fetch leaderboard:', error)
      // Return mock data for development
      return this.getMockLeaderboard()
    }
  }

  /**
   * Submit a new score to the leaderboard
   */
  async submitScore(request: SubmitScoreRequest): Promise<SubmitScoreResponse> {
    try {
      const formData = new FormData()
      formData.append('name', request.name)
      formData.append('score', request.score.toString())
      
      if (request.modelWeights) {
        const blob = new Blob([request.modelWeights], { type: 'application/octet-stream' })
        formData.append('modelWeights', blob, 'model.weights.bin')
      }

      const response = await fetch(`${this.baseUrl}/leaderboard`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.warn('Failed to submit score:', error)
      // Return mock response for development
      return {
        success: true,
        entry: {
          id: `mock-${Date.now()}`,
          name: request.name,
          score: request.score,
          createdAt: new Date().toISOString(),
        },
        isNewChampion: request.score > 10,
      }
    }
  }

  /**
   * Get the current champion model
   */
  async getChampion(): Promise<ChampionModel | null> {
    try {
      const response = await fetch(`${this.baseUrl}/champion`)
      if (!response.ok) {
        if (response.status === 404) return null
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      return await response.json()
    } catch (error) {
      console.warn('Failed to fetch champion:', error)
      return null
    }
  }

  /**
   * Download champion model weights
   */
  async downloadChampionWeights(): Promise<ArrayBuffer | null> {
    try {
      const champion = await this.getChampion()
      if (!champion) return null

      const response = await fetch(champion.weightsUrl)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      return await response.arrayBuffer()
    } catch (error) {
      console.warn('Failed to download champion weights:', error)
      return null
    }
  }

  /**
   * Mock leaderboard data for development
   */
  private getMockLeaderboard(): LeaderboardResponse {
    const mockEntries: LeaderboardEntry[] = [
      { id: '1', name: 'FlappyMaster', score: 42, createdAt: '2024-01-15T10:30:00Z', isChampion: true },
      { id: '2', name: 'DeepBird', score: 38, createdAt: '2024-01-14T15:45:00Z' },
      { id: '3', name: 'NeuralFlapper', score: 35, createdAt: '2024-01-13T09:20:00Z' },
      { id: '4', name: 'QAgent007', score: 31, createdAt: '2024-01-12T14:00:00Z' },
      { id: '5', name: 'BirdBrain', score: 28, createdAt: '2024-01-11T11:15:00Z' },
      { id: '6', name: 'PipeNavigator', score: 25, createdAt: '2024-01-10T16:30:00Z' },
      { id: '7', name: 'AirMaster', score: 22, createdAt: '2024-01-09T08:45:00Z' },
      { id: '8', name: 'SkyLearner', score: 19, createdAt: '2024-01-08T13:00:00Z' },
      { id: '9', name: 'WingNet', score: 15, createdAt: '2024-01-07T10:20:00Z' },
      { id: '10', name: 'FlapBot', score: 12, createdAt: '2024-01-06T17:00:00Z' },
    ]

    return {
      entries: mockEntries,
      champion: mockEntries[0],
    }
  }
}

// Export singleton instance
export const apiClient = new ApiClient()

// Export class for custom instantiation
export { ApiClient }




