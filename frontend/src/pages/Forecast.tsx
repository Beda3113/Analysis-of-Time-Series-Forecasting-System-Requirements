import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { forecastApi, ForecastPoint } from '../api/forecast';
import { trainingApi } from '../api/training';

const Forecast: React.FC = () => {
  const { id } = useParams();
  const [models, setModels] = useState<any[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [horizon, setHorizon] = useState(30);
  const [alpha, setAlpha] = useState(0.05);
  const [forecast, setForecast] = useState<ForecastPoint[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (id) {
      trainingApi.getModels(id).then(res => {
        setModels(res.data.items);
        if (res.data.items.length) setSelectedModel(res.data.items[0].id);
      }).catch(console.error);
    }
  }, [id]);

  const generateForecast = async () => {
    if (!id || !selectedModel) return;
    setLoading(true);
    setError(null);
    try {
      const response = await forecastApi.getForecast(id, { model_id: selectedModel, horizon, alpha });
      setForecast(response.data.predictions);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Ошибка прогнозирования');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>Прогнозирование</h1>
      <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
        {models.map(m => <option key={m.id} value={m.id}>{m.name} ({m.model_type}) - MAPE: {m.metrics?.mape}%</option>)}
      </select>
      <input type="number" value={horizon} onChange={(e) => setHorizon(parseInt(e.target.value))} min={1} max={365} />
      <select value={alpha} onChange={(e) => setAlpha(parseFloat(e.target.value))}>
        <option value={0.01}>99%</option>
        <option value={0.05}>95%</option>
        <option value={0.1}>90%</option>
      </select>
      <button onClick={generateForecast} disabled={loading}>{loading ? 'Загрузка...' : 'Построить прогноз'}</button>
      {error && <div className="error">{error}</div>}
      {forecast && (
        <table border={1}>
          <thead><tr><th>Шаг</th><th>Прогноз</th><th>Нижняя граница</th><th>Верхняя граница</th></tr></thead>
          <tbody>
            {forecast.map(p => <tr key={p.step}><td>{p.step}</td><td>{p.value.toFixed(2)}</td><td>{p.lower_bound?.toFixed(2)}</td><td>{p.upper_bound?.toFixed(2)}</td></tr>)}
          </tbody>
        </table>
      )}
      <Link to={`/series/${id}`}>Назад</Link>
    </div>
  );
};
export default Forecast;
