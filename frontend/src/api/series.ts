import { apiClient } from './client';

export interface Series {
  id: string;
  user_id: string;
  name: string;
  description: string | null;
  length: number;
  min_value: number;
  max_value: number;
  avg_value: number;
  created_at: string;
  updated_at: string;
}

export interface SeriesListResponse {
  items: Series[];
  total: number;
  page: number;
  page_size: number;
}

export const seriesApi = {
  upload: (file: File, name?: string) => {
    const formData = new FormData();
    formData.append('file', file);
    if (name) formData.append('name', name);
    return apiClient.post<{ series_id: string; name: string; length: number }>('/series/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  getAll: (params?: { page?: number; page_size?: number; search?: string }) =>
    apiClient.get<SeriesListResponse>('/series', { params }),
  getById: (id: string) => apiClient.get<Series>(`/series/${id}`),
  getPreview: (id: string, rows: number = 20) =>
    apiClient.get<{ headers: string[]; data: Array<{ index: number; date: string; value: number }> }>(
      `/series/${id}/preview`, { params: { rows } }
    ),
  update: (id: string, data: { name?: string; description?: string }) =>
    apiClient.patch<Series>(`/series/${id}`, data),
  delete: (id: string) => apiClient.delete(`/series/${id}`),
  getPlot: (id: string) => apiClient.get<string>(`/series/${id}/plot`),
};
