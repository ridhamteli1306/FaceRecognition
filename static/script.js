document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewImage = document.getElementById('preview-image');
    const uploadContent = document.querySelector('.upload-content');
    const predictBtn = document.getElementById('predict-btn');
    const resultsArea = document.getElementById('results-area');
    const resultsList = document.getElementById('results-list');
    const loading = document.getElementById('loading');

    // Drag and Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFiles);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles({ target: { files: files } });
    }

    function handleFiles(e) {
        const file = e.target.files[0];
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function (e) {
                previewImage.src = e.target.result;
                previewImage.style.display = 'block'; // Ensure block display
                uploadContent.classList.add('hidden');
                predictBtn.disabled = false;
                resultsArea.classList.add('hidden');
            }
            reader.readAsDataURL(file);
        }
    }

    predictBtn.addEventListener('click', async () => {
        const file = fileInput.files[0];
        if (!file) return;

        // UI Reset
        predictBtn.disabled = true;
        loading.classList.remove('hidden');
        resultsArea.classList.add('hidden');
        resultsList.innerHTML = '';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            loading.classList.add('hidden');
            predictBtn.disabled = false;

            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            if (data.status === 'no_faces') {
                resultsList.innerHTML = '<div class="result-item"><span class="celebrity-name">No faces detected</span></div>';
            } else if (data.status === 'success') {
                data.predictions.forEach(pred => {
                    const el = document.createElement('div');
                    el.className = `result-item ${pred.name === 'Unknown celebrity' ? 'unknown' : ''}`;
                    const percent = Math.round(pred.confidence * 100);

                    el.innerHTML = `
                        <div class="result-header">
                            <div class="result-info">
                                <span class="celebrity-name">${pred.name}</span>
                                <span class="confidence-text">${percent}%</span>
                            </div>
                            <div class="confidence-bar-bg">
                                <div class="confidence-bar-fill" style="width: ${percent}%"></div>
                            </div>
                        </div>
                    `;

                    if (pred.bio) {
                        const bioEl = document.createElement('div');
                        bioEl.className = 'bio-text';
                        bioEl.textContent = pred.bio;
                        el.appendChild(bioEl);
                    }

                    resultsList.appendChild(el);
                });
            }
            resultsArea.classList.remove('hidden');

        } catch (error) {
            console.error('Error:', error);
            loading.classList.add('hidden');
            predictBtn.disabled = false;
            alert('An error occurred during prediction.');
        }
    });
});
