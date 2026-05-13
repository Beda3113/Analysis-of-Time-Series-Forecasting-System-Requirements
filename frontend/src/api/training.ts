import { apiClient } from './client';

export interface TrainingRequest {
  model_type: 'xgboost' | 'lstm' | 'prophet' | 'sarima';
  name?: string;
  horizon: number;
  hyperparams?: Record<string, any>;
}

export interface TrainingResponse {
  task_id: string;
  model_id?: string;
  status: string;
  message: string;
}

export interface TrainingStatus {
  task_id: string;
  status: string;
  progress: number;
  result?: any;
  error?: string;
  updated_at: string;
}

export interface ModelInfo {
  id: string;
  series_id: string;
  user_id: string;
  model_type: string;
  name: string;
  hyperparams: Record<string, any>;
  metrics: { mae: number; rmse: number; mape: number };
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export const trainingApi = {
  start: (seriesId: string, data: TrainingRequest) =>
    apiClient.post<TrainingResponse>(`/training/${seriesId}`, data),
  getStatus: (taskId: string) => apiClient.get<TrainingStatus>(`/training/status/${taskId}`),
  getModels: (seriesId: string) => apiClient.get<{ items: ModelInfo[]; total: number }>('/training/models', {
    params: { series_id: seriesId },
  }),
  getModel: (modelId: string) => apiClient.get<ModelInfo>(`/training/models/${modelId}`),
  deleteModel: (modelId: string) => apiClient.delete(`/training/models/${modelId}`),
  activateModel: (modelId: string) => apiClient.post(`/training/models/${modelId}/activate`),
};
