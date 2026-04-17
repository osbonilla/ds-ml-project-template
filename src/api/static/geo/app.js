/* ─── Constantes ─────────────────────────────────────────────────────────── */
const API = '';

// Bounding box de California
const CA_BOUNDS = {
    latMin: 32.5, latMax: 42.0,
    lonMin: -124.5, lonMax: -114.1,
};

// Polígono simplificado de la costa de California
// Puntos clave para detectar si el clic cayó en el agua (Pacífico)
// Formato: [lat, lon] — línea costera aproximada de sur a norte
const CA_COAST_LINE = [
    [32.53, -117.12], [32.66, -117.24], [33.00, -117.28], [33.20, -117.38],
    [33.37, -117.59], [33.45, -117.65], [33.60, -117.87], [33.72, -118.03],
    [33.77, -118.20], [33.92, -118.42], [34.00, -118.51], [34.03, -118.52],
    [34.10, -119.08], [34.18, -119.22], [34.40, -119.73], [34.45, -119.83],
    [34.55, -120.07], [34.61, -120.19], [34.90, -120.46], [35.14, -120.64],
    [35.18, -120.74], [35.37, -120.86], [35.54, -121.00], [35.79, -121.33],
    [35.97, -121.51], [36.26, -121.83], [36.52, -121.93], [36.61, -121.89],
    [36.71, -121.80], [36.98, -122.03], [37.18, -122.40], [37.33, -122.40],
    [37.49, -122.51], [37.60, -122.50], [37.85, -122.48], [37.91, -122.68],
    [38.14, -122.88], [38.23, -122.98], [38.31, -123.07], [38.55, -123.25],
    [38.72, -123.44], [38.91, -123.68], [39.02, -123.69], [39.31, -123.80],
    [39.57, -123.76], [39.78, -123.82], [39.98, -124.07], [40.16, -124.24],
    [40.34, -124.33], [40.65, -124.20], [40.92, -124.15], [41.12, -124.14],
    [41.47, -124.07], [41.78, -124.18], [42.00, -124.21],
];

// Toast duration ms
const TOAST_DURATION = 3500;

/* ─── DOM ────────────────────────────────────────────────────────────────── */
const form        = document.getElementById('predict-form');
const submitBtn   = document.getElementById('submit-btn');
const btnText     = submitBtn.querySelector('.btn-text');
const btnLoading  = submitBtn.querySelector('.btn-loading');
const inputLat    = document.getElementById('latitude');
const inputLon    = document.getElementById('longitude');
const coordDisp   = document.getElementById('coord-display');

const stateIdle   = document.getElementById('state-idle');
const stateResult = document.getElementById('state-result');
const stateError  = document.getElementById('state-error');

const resultPrice = document.getElementById('result-price');
const resultBar   = document.getElementById('result-bar');
const errorMsg    = document.getElementById('error-msg');

const rLat = document.getElementById('r-lat');
const rLon = document.getElementById('r-lon');
const dRph = document.getElementById('d-rph');
const dBpr = document.getElementById('d-bpr');
const dPph = document.getElementById('d-pph');

const statusPill  = document.getElementById('status-pill');
const statusLabel = statusPill.querySelector('.status-label');
const mapToast    = document.getElementById('map-toast');

const PRICE_MAX = 500001;

/* ─── Estado inicial ─────────────────────────────────────────────────────── */
stateIdle.hidden   = false;
stateResult.hidden = true;
stateError.hidden  = true;

let selectedLat = null;
let selectedLon = null;
let marker      = null;
let toastTimer  = null;

/* ─── Mapa Leaflet ───────────────────────────────────────────────────────── */
const map = L.map('map', {
    center: [37.0, -119.5],
    zoom: 6,
    minZoom: 5,
    maxZoom: 13,
});

L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap contributors © CARTO',
    subdomains: 'abcd',
    maxZoom: 19,
}).addTo(map);

// Dibujar bounding box de California como referencia visual
const caBounds = [
    [CA_BOUNDS.latMin, CA_BOUNDS.lonMin],
    [CA_BOUNDS.latMax, CA_BOUNDS.lonMax],
];
L.rectangle(caBounds, {
    color: '#1a1a18',
    weight: 1,
    opacity: 0.15,
    fill: false,
    dashArray: '4 6',
}).addTo(map);

// Íconos de marcador
const iconOk = L.divIcon({
    className: '',
    html: `<div style="
        width:14px; height:14px; border-radius:50%;
        background:#1a6644; border:2px solid #fff;
        box-shadow:0 1px 4px rgba(0,0,0,0.25);
    "></div>`,
    iconSize: [14, 14],
    iconAnchor: [7, 7],
});

const iconError = L.divIcon({
    className: '',
    html: `<div style="
        width:14px; height:14px; border-radius:50%;
        background:#b83232; border:2px solid #fff;
        box-shadow:0 1px 4px rgba(0,0,0,0.25);
    "></div>`,
    iconSize: [14, 14],
    iconAnchor: [7, 7],
});

/* ─── Utilidades ─────────────────────────────────────────────────────────── */

function showToast(msg, type = 'warn') {
    clearTimeout(toastTimer);
    mapToast.textContent = msg;
    mapToast.className   = `map-toast ${type}`;
    mapToast.hidden      = false;
    toastTimer = setTimeout(() => { mapToast.hidden = true; }, TOAST_DURATION);
}

function isInCaBounds(lat, lon) {
    return lat >= CA_BOUNDS.latMin && lat <= CA_BOUNDS.latMax &&
           lon >= CA_BOUNDS.lonMin && lon <= CA_BOUNDS.lonMax;
}

/**
 * Determina si el punto (lat, lon) está en el Pacífico (al oeste de la línea costera).
 * Usa un ray casting simplificado comparando la longitud del punto con la longitud
 * costera interpolada para esa latitud.
 */
function isInOcean(lat, lon) {
    // Encontrar los dos puntos costeros que enmarcan la latitud del clic
    for (let i = 0; i < CA_COAST_LINE.length - 1; i++) {
        const [lat1, lon1] = CA_COAST_LINE[i];
        const [lat2, lon2] = CA_COAST_LINE[i + 1];

        if (lat >= Math.min(lat1, lat2) && lat <= Math.max(lat1, lat2)) {
            // Interpolar la longitud costera en esa latitud
            const t = (lat - lat1) / (lat2 - lat1);
            const coastLon = lon1 + t * (lon2 - lon1);
            // Si el clic está al oeste de la costa → océano
            return lon < coastLon;
        }
    }
    return false;
}

function setStatus(state, label) {
    statusPill.className    = 'status-pill ' + state;
    statusLabel.textContent = label;
}

function usd(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency', currency: 'USD', maximumFractionDigits: 0,
    }).format(value);
}

function animatePrice(target) {
    const duration = 800;
    const start    = performance.now();
    function tick(now) {
        const t    = Math.min((now - start) / duration, 1);
        const ease = 1 - Math.pow(1 - t, 3);
        resultPrice.textContent = usd(Math.round(target * ease));
        if (t < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}

function show(el) {
    stateIdle.hidden   = true;
    stateResult.hidden = true;
    stateError.hidden  = true;
    el.hidden = false;
}

/* ─── Clic en mapa ───────────────────────────────────────────────────────── */
map.on('click', (e) => {
    const { lat, lng } = e.latlng;
    const roundLat     = Math.round(lat * 10000) / 10000;
    const roundLon     = Math.round(lng * 10000) / 10000;

    // Validación 1: bounding box California
    if (!isInCaBounds(lat, lng)) {
        if (marker) marker.setIcon(iconError);
        showToast('⚠ Ubicación fuera de California. Seleccione un punto dentro del estado.');
        coordDisp.className = 'coord-display';
        submitBtn.disabled  = true;
        selectedLat = null;
        selectedLon = null;
        if (!marker) {
            marker = L.marker([roundLat, roundLon], { icon: iconError }).addTo(map);
        } else {
            marker.setLatLng([roundLat, roundLon]);
        }
        return;
    }

    // Validación 2: océano
    if (isInOcean(lat, lng)) {
        if (marker) marker.setIcon(iconError);
        showToast('⚠ La ubicación está en el océano. Seleccione un punto en tierra firme.');
        coordDisp.className = 'coord-display';
        submitBtn.disabled  = true;
        selectedLat = null;
        selectedLon = null;
        if (!marker) {
            marker = L.marker([roundLat, roundLon], { icon: iconError }).addTo(map);
        } else {
            marker.setLatLng([roundLat, roundLon]);
        }
        return;
    }

    // Válido — actualizar marcador y campos
    selectedLat = roundLat;
    selectedLon = roundLon;

    inputLat.value = roundLat;
    inputLon.value = roundLon;

    coordDisp.textContent = `${roundLat}° N,  ${roundLon}° W`;
    coordDisp.className   = 'coord-display active';

    mapToast.hidden    = true;
    submitBtn.disabled = false;

    if (!marker) {
        marker = L.marker([roundLat, roundLon], { icon: iconOk }).addTo(map);
    } else {
        marker.setLatLng([roundLat, roundLon]);
        marker.setIcon(iconOk);
    }

    marker.bindPopup(
        `<b>Ubicación seleccionada</b><br/>` +
        `Lat: ${roundLat}<br/>Lon: ${roundLon}`,
        { closeButton: false }
    ).openPopup();
});

/* ─── Health check ───────────────────────────────────────────────────────── */
async function checkHealth() {
    try {
        const res  = await fetch(`${API}/health`);
        const data = await res.json();
        setStatus(data.status === 'ok' ? 'ok' : 'error',
                  data.status === 'ok' ? 'Sistema operativo' : 'Modelo no disponible');
    } catch {
        setStatus('error', 'Sin conexión');
    }
}

checkHealth();
setInterval(checkHealth, 30000);

/* ─── Reset ──────────────────────────────────────────────────────────────── */
function reset() {
    show(stateIdle);
    resultBar.style.width = '0%';
    form.reset();
    submitBtn.disabled = true;
    inputLat.value     = '';
    inputLon.value     = '';
    coordDisp.textContent = 'Haga clic en el mapa para seleccionar ubicación';
    coordDisp.className   = 'coord-display';
    selectedLat = null;
    selectedLon = null;
    if (marker) { map.removeLayer(marker); marker = null; }
}

document.getElementById('reset-btn').addEventListener('click', reset);
document.getElementById('error-reset-btn').addEventListener('click', reset);

/* ─── Submit ─────────────────────────────────────────────────────────────── */
form.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (!selectedLat || !selectedLon) {
        showToast('⚠ Primero seleccione una ubicación en el mapa.');
        return;
    }

    const payload = {
        latitude           : selectedLat,
        longitude          : selectedLon,
        housing_median_age : parseFloat(document.getElementById('housing_median_age').value),
        total_rooms        : parseFloat(document.getElementById('total_rooms').value),
        total_bedrooms     : parseFloat(document.getElementById('total_bedrooms').value),
        population         : parseFloat(document.getElementById('population').value),
        households         : parseFloat(document.getElementById('households').value),
        median_income      : parseFloat(document.getElementById('median_income').value),
        ocean_proximity    : document.getElementById('ocean_proximity').value,
    };

    submitBtn.disabled   = true;
    btnText.hidden       = true;
    btnLoading.hidden    = false;

    try {
        const res  = await fetch(`${API}/predict/geo`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(payload),
        });

        const data = await res.json();

        if (!res.ok) {
            const detail = data.detail || `Error ${res.status}`;
            errorMsg.textContent = typeof detail === 'string' ? detail : JSON.stringify(detail);
            show(stateError);
        } else {
            const rph = (payload.total_rooms    / payload.households).toFixed(2);
            const bpr = (payload.total_bedrooms / payload.total_rooms).toFixed(4);
            const pph = (payload.population     / payload.households).toFixed(2);

            rLat.textContent = `${data.latitude}° N`;
            rLon.textContent = `${data.longitude}° W`;
            dRph.textContent = rph;
            dBpr.textContent = bpr;
            dPph.textContent = pph;

            show(stateResult);
            animatePrice(data.predicted_price);
            setTimeout(() => {
                resultBar.style.width =
                    Math.min((data.predicted_price / PRICE_MAX) * 100, 100) + '%';
            }, 80);

            // Actualizar popup del marcador con precio
            if (marker) {
                marker.bindPopup(
                    `<b>Estimación</b><br/>${usd(data.predicted_price)}`,
                    { closeButton: false }
                ).openPopup();
            }
        }

    } catch {
        errorMsg.textContent = 'No se pudo conectar con el servidor. Verifique que la aplicación esté en ejecución.';
        show(stateError);
    } finally {
        submitBtn.disabled = false;
        btnText.hidden     = false;
        btnLoading.hidden  = true;
    }
});