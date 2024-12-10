// dashboard/static/js/dashboard.js

document.addEventListener('DOMContentLoaded', () => {
    const socket = io();

    const logList = document.getElementById('log-list');
    const progressList = document.getElementById('progress-list');
    const downloadList = document.getElementById('download-list');
    const embeddingList = document.getElementById('embedding-list');
    const contentList = document.getElementById('content-list');
    const pipelineStatus = document.getElementById('pipeline-status');

    socket.on('connect', () => {
        appendLog('info', 'Connecté au serveur.');
    });

    socket.on('disconnect', () => {
        appendLog('error', 'Déconnecté du serveur.');
    });

    socket.on('log', (data) => {
        const level = data.level || 'info';
        const message = data.message || '';
        appendLog(level, message);
    });

    socket.on('download', (data) => {
        const fileType = data.file_type || 'Inconnu';
        const filename = data.filename || 'Non nommé';
        appendDownload(fileType, filename);
    });

    socket.on('progress', (data) => {
        appendProgress(data);
    });

    socket.on('embedding_processed', (data) => {
        const filename = data.filename || 'Non nommé';
        const chunk_id = data.chunk_id || 'N/A';
        appendEmbedding(filename, chunk_id);
    });

    socket.on('content_extracted', (data) => {
        const filename = data.filename || 'Non nommé';
        appendContentExtracted(filename);
    });

    socket.on('content_rewritten', (data) => {
        const filename = data.filename || 'Non nommé';
        appendContentRewritten(filename);
    });

    socket.on('crawl_completed', (data) => {
        const duration = data.duration_seconds || 0;
        const status = data.status || 'inconnu';
        pipelineStatus.textContent = `Terminé en ${duration.toFixed(2)} secondes avec le statut : ${status}`;
        appendLog('info', `Crawl terminé en ${duration.toFixed(2)} secondes avec le statut : ${status}`);
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
            message = `Nouvelle URL découverte : ${data.url}`;
        } else if (data.type === 'page_crawled') {
            message = `Page crawled : ${data.url}`;
        } else if (data.type === 'content_processed') {
            message = `Contenu traité pour la page ${data.page_num} du document ${data.document}`;
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

    function appendEmbedding(filename, chunk_id) {
        const li = document.createElement('li');
        li.textContent = `Embedding traité : ${filename}, Chunk ID : ${chunk_id}`;
        embeddingList.prepend(li);
    }

    function appendContentExtracted(filename) {
        const li = document.createElement('li');
        li.textContent = `Contenu extrait : ${filename}`;
        contentList.prepend(li);
    }

    function appendContentRewritten(filename) {
        const li = document.createElement('li');
        li.textContent = `Contenu réécrit : ${filename}`;
        contentList.prepend(li);
    }
});
