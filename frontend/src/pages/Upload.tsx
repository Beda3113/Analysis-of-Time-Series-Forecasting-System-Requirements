import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { seriesApi } from '../api/series';

const Upload: React.FC = () => {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState('');
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      if (!name) setName(selected.name.replace(/\.[^/.]+$/, ''));
    }
  };

  const handleUpload = async () => {
    if (!file) return alert('Выберите файл');
    setUploading(true);
    setError(null);
    try {
      const response = await seriesApi.upload(file, name);
      alert(`Ряд "${response.data.name}" загружен! (${response.data.length} точек)`);
      navigate('/dashboard');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Ошибка загрузки');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-container">
      <h1>Загрузка временного ряда</h1>
      <input type="file" accept=".csv,.xlsx,.xls" onChange={handleFileChange} />
      {file && (
        <>
          <input type="text" placeholder="Название ряда" value={name} onChange={(e) => setName(e.target.value)} />
          <p>Файл: {file.name} ({(file.size / 1024).toFixed(2)} KB)</p>
          <button onClick={handleUpload} disabled={uploading}>{uploading ? 'Загрузка...' : 'Загрузить'}</button>
        </>
      )}
      {error && <div className="error">{error}</div>}
    </div>
  );
};
export default Upload;
