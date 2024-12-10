// dashboard/static/js/dashboard.js

document.addEventListener('DOMContentLoaded', () => {
    const socket = io();

    const logList = document.getElementById('log-list');
    const progressList = document.getElementById('progress-list');
    const downloadList = document.getElementById('download-list');
    const pipelineStatus = document.getElementById('pipeline-status');

    socket.on('connect', () => {
        appendLog('info', 'Connected to the server.');
    });

    socket.on('disconnect', () => {
        appendLog('error', 'Disconnected from the server.');
    });

    socket.on('log', (data) => {
        const level = data.level || 'info';
        const message = data.message || '';
        appendLog(level, message);
    });

    socket.on('download', (data) => {
        const fileType = data.file_type || 'Unknown';
        const filename = data.filename || 'Unnamed';
        appendDownload(fileType, filename);
    });

    socket.on('progress', (data) => {
        appendProgress(data);
    });

    socket.on('crawl_completed', (data) => {
        const duration = data.duration_seconds || 0;
        const status = data.status || 'unknown';
        pipelineStatus.textContent = `Completed in ${duration.toFixed(2)} seconds with status: ${status}`;
        appendLog('info', `Crawl completed in ${duration.toFixed(2)} seconds with status: ${status}`);
    });

    function appendLog(level, message) {
        const li = document.createElement('li');
        li.className = level;
        li.textContent = `[${level.toUpperCase()}] ${message}`;
        logList.prepend(li);
    }

    function appendProgress(data) {
        let message = '';
        if (data.type === 'new_url') {
            message = `New URL discovered: ${data.url}`;
        } else if (data.type === 'page_crawled') {
            message = `Page crawled: ${data.url}`;
        } else if (data.type === 'content_extracted') {
            message = `Content extracted from: ${data.url}, saved as ${data.filename}`;
        }
        if (message) {
            const li = document.createElement('li');
            li.textContent = message;
            progressList.prepend(li);
        }
    }

    function appendDownload(fileType, filename) {
        const li = document.createElement('li');
        li.textContent = `[${fileType}] ${filename}`;
        downloadList.prepend(li);
    }
});
