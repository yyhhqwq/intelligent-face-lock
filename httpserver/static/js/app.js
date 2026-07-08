const state = {
    currentPage: 1,
    perPage: 10,
    deviceFilter: '',
    statusFilter: '',
    dateFilter: '',
    searchQuery: '',
    refreshInterval: null,
};

async function fetchDashboard() {
    try {
        const resp = await fetch('/api/dashboard');
        const data = await resp.json();

        document.getElementById('today-traffic').textContent = data.today_traffic.toLocaleString();
        document.getElementById('online-devices').textContent = data.online_devices;
        document.getElementById('total-devices').textContent = `共 ${data.total_devices} 台设备`;
        document.getElementById('liveness-rate').textContent = data.liveness_rate + '%';
        document.getElementById('stranger-alerts').textContent = data.stranger_alerts;

        const trafficTrend = document.getElementById('traffic-trend');
        if (data.traffic_trend >= 0) {
            trafficTrend.className = 'stat-trend positive';
            trafficTrend.textContent = `↑ +${data.traffic_trend}%`;
        } else {
            trafficTrend.className = 'stat-trend negative';
            trafficTrend.textContent = `↓ ${data.traffic_trend}%`;
        }

        const livenessTrend = document.getElementById('liveness-trend');
        if (data.liveness_trend >= 0) {
            livenessTrend.className = 'stat-trend positive';
            livenessTrend.textContent = `↑ +${data.liveness_trend}%`;
        } else {
            livenessTrend.className = 'stat-trend negative';
            livenessTrend.textContent = `↓ ${data.liveness_trend}%`;
        }
    } catch (e) {
        console.error('Failed to fetch dashboard:', e);
    }
}

async function fetchDevices() {
    try {
        const resp = await fetch('/api/devices');
        const devices = await resp.json();
        const select = document.getElementById('filter-device');
        const currentVal = select.value;
        select.innerHTML = '<option value="">全部设备</option>';
        devices.forEach(d => {
            const opt = document.createElement('option');
            opt.value = d.device_id;
            opt.textContent = `${d.name || d.device_id}`;
            select.appendChild(opt);
        });
        select.value = currentVal;
    } catch (e) {
        console.error('Failed to fetch devices:', e);
    }
}

function getEventTagClass(eventType) {
    if (eventType.includes('进出') || eventType.includes('门')) return 'Personnel';
    if (eventType.includes('陌生人')) return 'Stranger';
    if (eventType.includes('活体')) return 'Liveness';
    if (eventType.includes('性能')) return 'Performance';
    return 'Personnel';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function fetchEvents() {
    try {
        const params = new URLSearchParams({
            page: state.currentPage,
            per_page: state.perPage,
        });
        if (state.deviceFilter) params.set('device_id', state.deviceFilter);
        if (state.statusFilter) params.set('status', state.statusFilter);
        if (state.dateFilter) params.set('date', state.dateFilter);

        const resp = await fetch(`/api/events?${params}`);
        const data = await resp.json();

        renderTable(data);
        renderPagination(data);
    } catch (e) {
        console.error('Failed to fetch events:', e);
    }
}

function renderTable(data) {
    const tbody = document.getElementById('events-tbody');
    if (!data.items || data.items.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6">
                    <div class="empty-state">
                        <div class="icon">📭</div>
                        <div>暂无数据</div>
                    </div>
                </td>
            </tr>
        `;
        return;
    }

    let filteredItems = data.items;
    if (state.searchQuery) {
        const q = state.searchQuery.toLowerCase();
        filteredItems = filteredItems.filter(item =>
            item.device_id.toLowerCase().includes(q) ||
            item.device_name.toLowerCase().includes(q) ||
            item.details.toLowerCase().includes(q)
        );
    }

    tbody.innerHTML = filteredItems.map(item => `
        <tr>
            <td>${escapeHtml(item.device_id)}</td>
            <td>${escapeHtml(item.device_name)}</td>
            <td><span class="event-tag ${getEventTagClass(item.event_type)}">${escapeHtml(item.event_type)}</span></td>
            <td>${escapeHtml(item.details)}</td>
            <td>${escapeHtml(item.time)}</td>
            <td>
                <span class="status-badge ${item.status}">
                    <span class="status-dot ${item.status}"></span>
                    ${item.status === 'success' ? '成功' : '异常'}
                </span>
            </td>
        </tr>
    `).join('');
}

function renderPagination(data) {
    const info = document.getElementById('pagination-info');
    info.textContent = `共 ${data.total} 条记录，第 ${data.page} / ${data.total_pages || 1} 页`;

    const container = document.getElementById('pagination');
    if (data.total_pages <= 1) {
        container.innerHTML = '';
        return;
    }

    let html = '';
    html += `<button class="page-btn" onclick="goToPage(${data.page - 1})" ${data.page <= 1 ? 'disabled' : ''}>&lt;</button>`;

    const maxVisible = 5;
    let start = Math.max(1, data.page - Math.floor(maxVisible / 2));
    let end = Math.min(data.total_pages, start + maxVisible - 1);
    if (end - start < maxVisible - 1) {
        start = Math.max(1, end - maxVisible + 1);
    }

    for (let i = start; i <= end; i++) {
        html += `<button class="page-btn ${i === data.page ? 'active' : ''}" onclick="goToPage(${i})">${i}</button>`;
    }

    html += `<button class="page-btn" onclick="goToPage(${data.page + 1})" ${data.page >= data.total_pages ? 'disabled' : ''}>&gt;</button>`;
    container.innerHTML = html;
}

function goToPage(page) {
    state.currentPage = page;
    fetchEvents();
}

function exportCSV() {
    const params = new URLSearchParams();
    if (state.deviceFilter) params.set('device_id', state.deviceFilter);
    if (state.statusFilter) params.set('status', state.statusFilter);
    if (state.dateFilter) params.set('date', state.dateFilter);
    window.open(`/api/export?${params}`, '_blank');
}

function setupFilters() {
    document.getElementById('filter-device').addEventListener('change', e => {
        state.deviceFilter = e.target.value;
        state.currentPage = 1;
        fetchEvents();
    });

    document.getElementById('filter-status').addEventListener('change', e => {
        state.statusFilter = e.target.value;
        state.currentPage = 1;
        fetchEvents();
    });

    document.getElementById('filter-date').addEventListener('change', e => {
        state.dateFilter = e.target.value;
        state.currentPage = 1;
        fetchEvents();
    });

    document.getElementById('search-input').addEventListener('input', e => {
        state.searchQuery = e.target.value;
        fetchEvents();
    });
}

function startAutoRefresh() {
    fetchDashboard();
    fetchDevices();
    fetchEvents();
    state.refreshInterval = setInterval(() => {
        fetchDashboard();
        fetchEvents();
    }, 5000);
}

document.addEventListener('DOMContentLoaded', () => {
    setupFilters();
    startAutoRefresh();
});
