/* ─── Constantes ─────────────────────────────────────────────────────────── */
const API = '';

const TOAST_DURATION = 3500;
const PRICE_MAX      = 500001;

/* ─── GeoJSON de California ──────────────────────────────────────────────── */
const CA_GEOJSON_URL = '/geo/static/california.geojson';
let   caPolygons     = [];

async function loadCaliforniaGeoJSON() {
    const res  = await fetch(CA_GEOJSON_URL);
    const data = await res.json();
    data.features.forEach(feature => {
        const geom = feature.geometry;
        if (geom.type === 'Polygon') {
            caPolygons.push(geom.coordinates[0]);
        } else if (geom.type === 'MultiPolygon') {
            geom.coordinates.forEach(poly => caPolygons.push(poly[0]));
        }
    });
}

function pointInRing(lon, lat, ring) {
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
        const [xi, yi] = ring[i];
        const [xj, yj] = ring[j];
        const intersect = ((yi > lat) !== (yj > lat)) &&
                          (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

function isInCalifornia(lat, lon) {
    return caPolygons.some(ring => pointInRing(lon, lat, ring));
}

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
    center  : [37.0, -119.5],
    zoom    : 6,
    minZoom : 5,
    maxZoom : 13,
});

L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap contributors © CARTO',
    subdomains : 'abcd',
    maxZoom    : 19,
}).addTo(map);

async function addCaliforniaLayer() {
    const res  = await fetch(CA_GEOJSON_URL);
    const data = await res.json();
    L.geoJSON(data, {
        style: { color: '#1a1a18', weight: 1.2, opacity: 0.25, fill: false }
    }).addTo(map);
}

const iconOk = L.divIcon({
    className: '',
    html: `<div style="width:14px;height:14px;border-radius:50%;background:#1a6644;border:2px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,0.25);"></div>`,
    iconSize: [14, 14], iconAnchor: [7, 7],
});

const iconError = L.divIcon({
    className: '',
    html: `<div style="width:14px;height:14px;border-radius:50%;background:#b83232;border:2px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,0.25);"></div>`,
    iconSize: [14, 14], iconAnchor: [7, 7],
});

/* ─── Utilidades ─────────────────────────────────────────────────────────── */
function showToast(msg, type = 'warn') {
    clearTimeout(toastTimer);
    mapToast.textContent = msg;
    mapToast.className   = `map-toast ${type}`;
    mapToast.hidden      = false;
    toastTimer = setTimeout(() => { mapToast.hidden = true; }, TOAST_DURATION);
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

function placeMarker(lat, lon, valid) {
    if (!marker) {
        marker = L.marker([lat, lon], { icon: valid ? iconOk : iconError }).addTo(map);
    } else {
        marker.setLatLng([lat, lon]);
        marker.setIcon(valid ? iconOk : iconError);
    }
}

/* ─── Clic en mapa ───────────────────────────────────────────────────────── */
map.on('click', (e) => {
    const { lat, lng } = e.latlng;
    const roundLat     = Math.round(lat * 10000) / 10000;
    const roundLon     = Math.round(lng * 10000) / 10000;

    placeMarker(roundLat, roundLon, false);

    if (!isInCalifornia(lat, lng)) {
        showToast('⚠ Ubicación fuera de California o en el océano. Seleccione un punto en tierra firme dentro del estado.');
        coordDisp.className = 'coord-display';
        submitBtn.disabled  = true;
        selectedLat = null;
        selectedLon = null;
        return;
    }

    selectedLat = roundLat;
    selectedLon = roundLon;
    inputLat.value        = roundLat;
    inputLon.value        = roundLon;
    coordDisp.textContent = `${roundLat}° N,  ${roundLon}° W`;
    coordDisp.className   = 'coord-display active';
    mapToast.hidden       = true;
    submitBtn.disabled    = false;

    placeMarker(roundLat, roundLon, true);
    marker.bindPopup(
        `<b>Ubicación seleccionada</b><br/>Lat: ${roundLat}<br/>Lon: ${roundLon}`,
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
    submitBtn.disabled    = true;
    inputLat.value        = '';
    inputLon.value        = '';
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

    submitBtn.disabled = true;
    btnText.hidden     = true;
    btnLoading.hidden  = false;

    try {
        const res  = await fetch(`${API}/predict/geo`, {
            method : 'POST',
            headers: { 'Content-Type': 'application/json' },
            body   : JSON.stringify(payload),
        });

        const data = await res.json();

        if (!res.ok) {
            const detail = data.detail || `Error ${res.status}`;
            errorMsg.textContent = typeof detail === 'string' ? detail : JSON.stringify(detail);
            show(stateError);
        } else {
            rLat.textContent = `${data.latitude}° N`;
            rLon.textContent = `${data.longitude}° W`;
            dRph.textContent = (payload.total_rooms    / payload.households).toFixed(2);
            dBpr.textContent = (payload.total_bedrooms / payload.total_rooms).toFixed(4);
            dPph.textContent = (payload.population     / payload.households).toFixed(2);

            show(stateResult);
            animatePrice(data.predicted_price);
            setTimeout(() => {
                resultBar.style.width =
                    Math.min((data.predicted_price / PRICE_MAX) * 100, 100) + '%';
            }, 80);

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

/* ─── Inicialización ─────────────────────────────────────────────────────── */
(async () => {
    await loadCaliforniaGeoJSON();
    await addCaliforniaLayer();
})();