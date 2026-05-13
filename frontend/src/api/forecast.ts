import { apiClient } from './client';

export interface ForecastPoint {
  step: number;
  value: number;
  lower_bound?: number;
  upper_bound?: number;
  date?: string;
}

export interface ForecastResponse {
  series_id: string;
  model_id: string;
  model_type: string;
  model_name: string;
  horizon: number;
  alpha: number;
  created_at: string;
  historical_values?: number[];
  predictions: ForecastPoint[];
  metrics?: Record<string, number>;
}

export const forecastApi = {
  getForecast: (seriesId: string, params?: { model_id?: string; horizon?: number; alpha?: number }) =>
    apiClient.get<ForecastResponse>(`/forecast/${seriesId}`, { params }),
  exportCsv: (seriesId: string, params?: { model_id?: string; horizon?: number }) =>
    apiClient.get(`/forecast/${seriesId}/export`, { params, responseType: 'blob' }),
  getMetrics: (modelId: string) => apiClient.get<{ metrics: Record<string, number> }>(`/forecast/metrics/${modelId}`),
};
