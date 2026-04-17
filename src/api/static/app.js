const API = '';

const form       = document.getElementById('predict-form');
const submitBtn  = document.getElementById('submit-btn');
const btnText    = submitBtn.querySelector('.btn-text');
const btnLoading = submitBtn.querySelector('.btn-loading');

const stateIdle   = document.getElementById('state-idle');
const stateResult = document.getElementById('state-result');
const stateError  = document.getElementById('state-error');

const resultPrice = document.getElementById('result-price');
const resultBar   = document.getElementById('result-bar');
const errorMsg    = document.getElementById('error-msg');

const dRph = document.getElementById('d-rph');
const dBpr = document.getElementById('d-bpr');
const dPph = document.getElementById('d-pph');

const statusPill  = document.getElementById('status-pill');
const statusLabel = statusPill.querySelector('.status-label');

const PRICE_MAX = 500001;

// Estado inicial — siempre arranca en idle sin importar el HTML
stateIdle.hidden   = false;
stateResult.hidden = true;
stateError.hidden  = true;

function usd(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency', currency: 'USD', maximumFractionDigits: 0
    }).format(value);
}

function animatePrice(target) {
    const duration = 800;
    const start    = performance.now();
    function tick(now) {
        const t = Math.min((now - start) / duration, 1);
        const ease = 1 - Math.pow(1 - t, 3);
        resultPrice.textContent = usd(Math.round(target * ease));
        if (t < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}

function show(el) {
    [stateIdle, stateResult, stateError].forEach(s => s.hidden = true);
    el.hidden = false;
}

function setStatus(state, label) {
    statusPill.className = 'status-pill ' + state;
    statusLabel.textContent = label;
}

async function checkHealth() {
    try {
        const res  = await fetch(`${API}/health`);
        const data = await res.json();
        if (data.status === 'ok') {
            setStatus('ok', 'Sistema operativo');
        } else {
            setStatus('error', 'Modelo no disponible');
        }
    } catch {
        setStatus('error', 'Sin conexión');
    }
}

checkHealth();
setInterval(checkHealth, 30000);

document.getElementById('reset-btn').addEventListener('click', reset);
document.getElementById('error-reset-btn').addEventListener('click', reset);

function reset() {
    show(stateIdle);
    resultBar.style.width = '0%';
    form.reset();
    submitBtn.disabled = false;
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const payload = {
        housing_median_age: parseFloat(document.getElementById('housing_median_age').value),
        total_rooms:        parseFloat(document.getElementById('total_rooms').value),
        total_bedrooms:     parseFloat(document.getElementById('total_bedrooms').value),
        population:         parseFloat(document.getElementById('population').value),
        households:         parseFloat(document.getElementById('households').value),
        median_income:      parseFloat(document.getElementById('median_income').value),
        ocean_proximity:    document.getElementById('ocean_proximity').value,
    };

    submitBtn.disabled   = true;
    btnText.hidden       = true;
    btnLoading.hidden    = false;

    try {
        const res  = await fetch(`${API}/predict`, {
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

            dRph.textContent = rph;
            dBpr.textContent = bpr;
            dPph.textContent = pph;

            show(stateResult);
            animatePrice(data.predicted_price);
            setTimeout(() => {
                resultBar.style.width = Math.min((data.predicted_price / PRICE_MAX) * 100, 100) + '%';
            }, 80);
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