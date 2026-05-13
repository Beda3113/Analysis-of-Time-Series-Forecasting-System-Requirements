import { Routes, Route, Navigate } from 'react-router-dom';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import Upload from './pages/Upload';
import SeriesDetail from './pages/SeriesDetail';
import Training from './pages/Training';
import Forecast from './pages/Forecast';
import Interpretation from './pages/Interpretation';
import Preprocessing from './pages/Preprocessing';
import { PrivateRoute } from './components/common/PrivateRoute';

function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route element={<PrivateRoute />}>
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/series/:id" element={<SeriesDetail />} />
        <Route path="/series/:id/training" element={<Training />} />
        <Route path="/series/:id/forecast" element={<Forecast />} />
        <Route path="/series/:id/interpretation" element={<Interpretation />} />
        <Route path="/series/:id/preprocessing" element={<Preprocessing />} />
      </Route>
      <Route path="/" element={<Navigate to="/dashboard" replace />} />
    </Routes>
  );
}

export default App;
