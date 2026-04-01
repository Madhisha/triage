import { NavLink } from 'react-router-dom'

export default function Navbar({ dark, onToggleTheme }) {
  return (
    <nav className="navbar" id="main-navbar">
      <div className="navbar__brand">
        <span className="navbar__icon">⚕</span>
        <span className="navbar__title">Triage Command</span>
      </div>

      <div className="navbar__links">
        <NavLink
          to="/"
          end
          className={({ isActive }) =>
            `navbar__link${isActive ? ' navbar__link--active' : ''}`
          }
          id="nav-triage"
        >
          <span className="navbar__link-icon">📋</span>
          Assessment
        </NavLink>
        <NavLink
          to="/monitor"
          className={({ isActive }) =>
            `navbar__link${isActive ? ' navbar__link--active' : ''}`
          }
          id="nav-monitor"
        >
          <span className="navbar__link-icon">📡</span>
          Live Monitor
          <span className="navbar__live-dot" />
        </NavLink>
      </div>

      <button className="theme-toggle" onClick={onToggleTheme} id="theme-toggle">
        {dark ? '☀ Light' : '☾ Dark'}
      </button>
    </nav>
  )
}
