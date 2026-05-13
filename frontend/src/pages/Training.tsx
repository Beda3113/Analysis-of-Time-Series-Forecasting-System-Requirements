import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { trainingApi, TrainingRequest } from '../api/training';
import { seriesApi } from '../api/series';

const Training: React.FC = () => {
  const { id } = useParams();
  const [modelType, setModelType] = useState<'xgboost' | 'lstm' | 'prophet' | 'sarima'>('xgboost');
  const [horizon, setHorizon] = useState(30);
  const [name, setName] = useState('');
  const [training, setTraining] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [modelId, setModelId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [seriesName, setSeriesName] = useState('');

  useEffect(() => {
    if (id) {
      seriesApi.getById(id).then(res => setSeriesName(res.data.name)).catch(console.error);
    }
  }, [id]);

  useEffect(() => {
    if (!taskId) return;
    const interval = setInterval(async () => {
      try {
        const status = await trainingApi.getStatus(taskId);
        setProgress(status.data.progress);
        if (status.data.status === 'completed') {
          setTraining(false);
          setModelId(status.data.result?.model_id);
          clearInterval(interval);
        } else if (status.data.status === 'failed') {
          setError(status.data.error || 'Ошибка обучения');
          setTraining(false);
          clearInterval(interval);
        }
      } catch (err) {
        console.error(err);
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [taskId]);

  const startTraining = async () => {
    if (!id) return;
    setTraining(true);
    setError(null);
    try {
      const request: TrainingRequest = { model_type: modelType, horizon, name: name || undefined };
      const response = await trainingApi.start(id, request);
      setTaskId(response.data.task_id);
      if (response.data.model_id) setModelId(response.data.model_id);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Ошибка запуска обучения');
      setTraining(false);
    }
  };

  return (
    <div>
      <h1>Обучение модели: {seriesName}</h1>
      <select value={modelType} onChange={(e) => setModelType(e.target.value as any)}>
        <option value="xgboost">XGBoost</option>
        <option value="lstm">LSTM</option>
        <option value="prophet">Prophet</option>
        <option value="sarima">SARIMA</option>
      </select>
      <input type="number" value={horizon} onChange={(e) => setHorizon(parseInt(e.target.value))} min={1} max={365} />
      <input type="text" placeholder="Название модели" value={name} onChange={(e) => setName(e.target.value)} />
      <button onClick={startTraining} disabled={training}>{training ? `Обучение... ${progress}%` : 'Начать обучение'}</button>
      {error && <div className="error">{error}</div>}
      {modelId && <div>Модель обучена! ID: {modelId}. <Link to={`/series/${id}/forecast`}>Перейти к прогнозу</Link></div>}
    </div>
  );
};
export default Training;
