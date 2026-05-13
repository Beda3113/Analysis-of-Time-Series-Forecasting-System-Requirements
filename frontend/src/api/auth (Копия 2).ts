import { apiClient } from './client';

export interface User {
  id: string;
  email: string;
  name: string;
  is_active: boolean;
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

export const authApi = {
  register: (email: string, password: string, name: string) =>
    apiClient.post<AuthResponse>('/auth/register', { email, password, name }),
  login: (email: string, password: string) =>
    apiClient.post<AuthResponse>('/auth/login', { email, password }),
  refresh: (refreshToken: string) =>
    apiClient.post<{ access_token: string }>('/auth/refresh', { refresh_token: refreshToken }),
  logout: (refreshToken: string) =>
    apiClient.post('/auth/logout', { refresh_token: refreshToken }),
  getMe: () => apiClient.get<User>('/auth/me'),
};
