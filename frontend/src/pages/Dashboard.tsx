import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { seriesApi, Series } from '../api/series';

const Dashboard: React.FC = () => {
  const [series, setSeries] = useState<Series[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadSeries();
  }, [search]);

  const loadSeries = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await seriesApi.getAll({ search, page: 1, page_size: 50 });
      setSeries(response.data.items);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Ошибка загрузки данных');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: string, name: string) => {
    if (!confirm(`Удалить ряд "${name}"?`)) return;
    try {
      await seriesApi.delete(id);
      setSeries(series.filter(s => s.id !== id));
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Ошибка удаления');
    }
  };

  if (loading) return <div className="loading">Загрузка...</div>;
  if (error) return <div className="error">{error}</div>;

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <h1>Мои временные ряды</h1>
        <Link to="/upload" className="upload-button">+ Загрузить ряд</Link>
      </div>
      <div className="search-bar">
        <input type="text" placeholder="Поиск..." value={search} onChange={(e) => setSearch(e.target.value)} />
      </div>
      {series.length === 0 ? (
        <div className="empty-state">Нет загруженных рядов</div>
      ) : (
        <div className="series-grid">
          {series.map(s => (
            <div key={s.id} className="series-card">
              <h3>{s.name}</h3>
              <p>Точек: {s.length} | Диапазон: {s.min_value?.toFixed(2)} - {s.max_value?.toFixed(2)}</p>
              <div className="card-actions">
                <Link to={`/series/${s.id}`}>Детали</Link>
                <Link to={`/series/${s.id}/training`}>Обучить</Link>
                <Link to={`/series/${s.id}/forecast`}>Прогноз</Link>
                <button onClick={() => handleDelete(s.id, s.name)}>Удалить</button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
export default Dashboard;
