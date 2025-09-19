/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI, Modality, Part, GenerateContentResponse } from '@google/genai';

// --- CONSTANTS ---
const GEMINI_MODEL = 'gemini-2.5-flash-image-preview';
const REFERENCE_REGEX = /@(\w+)/g;
const KEY_REGEX = /^\w+$/;

// --- STATE ---
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
const references = new Map<string, { data: string; mimeType: string }>();
let stagedImage: {
    file: File;
    data: string;
    mimeType: string;
    maskData?: string; // The pure black-and-white mask for the API
    maskPreviewData?: string; // The composite image for the UI preview
} | null = null;
let imageToSave: { data: string; mimeType: string } | null = null;
let saveModalSource: 'upload' | 'generated' | null = null;

// Editor State
let isDrawing = false;
let originalImage: HTMLImageElement | null = null;
let currentTool = 'brush';
let canvasHistory: ImageData[] = [];
let shapeStartX = 0;
let shapeStartY = 0;


// --- DOM ELEMENTS ---
const chatHistory = document.getElementById('chat-history')!;
const loadingIndicator = document.getElementById('loading-indicator')!;
const imagePreviewContainer = document.getElementById('image-preview-container')!;
const imagePreview = document.getElementById('image-preview') as HTMLImageElement;
const editMaskBtn = document.getElementById('edit-mask-btn')!;
const removeImageBtn = document.getElementById('remove-image-btn')!;
const chatForm = document.getElementById('chat-form')!;
const uploadBtn = document.getElementById('upload-btn')! as HTMLButtonElement;
const imageUpload = document.getElementById('image-upload') as HTMLInputElement;
const promptInput = document.getElementById('prompt-input') as HTMLTextAreaElement;
const sendBtn = document.getElementById('send-btn')! as HTMLButtonElement;

// Reference Picker Elements
const refPickerBtn = document.getElementById('ref-picker-btn')!;
const referencePickerPopover = document.getElementById('reference-picker-popover')!;
const referenceSearchInput = document.getElementById('reference-search-input') as HTMLInputElement;
const referencePickerGrid = document.getElementById('reference-picker-grid')!;

// Editor Modal Elements
const editorModal = document.getElementById('editor-modal')!;
const editorCloseBtn = editorModal.querySelector('.close-btn')!;
const brushSize = document.getElementById('brush-size') as HTMLInputElement;
const editorCanvas = document.getElementById('editor-canvas') as HTMLCanvasElement;
const saveMaskBtn = document.getElementById('save-mask-btn')!;
const toolSelection = document.getElementById('tool-selection')!;
const colorPalette = document.getElementById('color-palette')!;
const undoBtn = document.getElementById('undo-btn')!;
const ctx = editorCanvas.getContext('2d')!;

// Save Key Modal Elements
const saveKeyModal = document.getElementById('save-key-modal')!;
const saveKeyModalCloseBtn = saveKeyModal.querySelector('.close-btn')!;
const saveKeyPreview = document.getElementById('save-key-preview') as HTMLImageElement;
const referenceKeyInput = document.getElementById('reference-key-input') as HTMLInputElement;
const keyErrorMessage = document.getElementById('key-error-message')!;
const saveKeyBtn = document.getElementById('save-key-btn')! as HTMLButtonElement;
const useOnceBtn = document.getElementById('use-once-btn')! as HTMLButtonElement;

// References Management Modal Elements
const manageReferencesBtn = document.getElementById('manage-references-btn')!;
const referencesModal = document.getElementById('references-modal')!;
const referencesModalCloseBtn = referencesModal.querySelector('.close-btn')!;
const referencesList = document.getElementById('references-list')!;


// --- CORE LOGIC ---

/**
 * Handles form submission, sending the prompt and any staged image to the AI.
 */
async function handleFormSubmit(e: Event) {
    e.preventDefault();
    const prompt = promptInput.value.trim();

    if (prompt.toLowerCase() === 'help') {
        showHelpMessage();
        resetInputArea();
        return;
    }

    if (!prompt && !stagedImage) return;

    const userImageData = stagedImage?.maskPreviewData || stagedImage?.data;
    addMessage('user', prompt, userImageData ? [userImageData] : []);

    const imageToSend = stagedImage;
    resetInputArea();
    toggleLoading(true);

    try {
        await callGeminiApi(prompt, imageToSend);
    } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown error';
        console.error('Error processing request:', error);
        addMessage('ai', `An error occurred: ${message}`);
    } finally {
        toggleLoading(false);
    }
}

/**
 * Validates references, builds the request, and calls the Gemini API.
 */
async function callGeminiApi(prompt: string, image: typeof stagedImage | null) {
    if (!prompt && !image) {
        addMessage('ai', 'Please provide a prompt or an image.');
        return;
    }

    // 1. Validate references in the prompt
    const { referencedKeys, missingKeys } = parsePromptForReferences(prompt);
    if (missingKeys.length > 0) {
        addMessage('ai', `Error: The following reference keys were not found: ${missingKeys.join(', ')}.`);
        return;
    }

    // 2. Build the parts array for the API call
    const parts = buildApiParts(prompt, image, referencedKeys);
    if (parts.length === 0) return;

    // 3. Call the API and process the response
    try {
        const response = await ai.models.generateContent({
            model: GEMINI_MODEL,
            contents: { parts },
            config: {
                responseModalities: [Modality.IMAGE, Modality.TEXT],
            },
        });
        processApiResponse(response);
    } catch (error) {
        const message = error instanceof Error ? error.message : 'Please try again.';
        console.error('API Error:', error);
        addMessage('ai', `Sorry, an error occurred. ${message}`);
    }
}

/**
 * Parses a prompt string to find and validate reference keys.
 * @returns An object containing the set of valid keys and an array of missing keys.
 */
function parsePromptForReferences(prompt: string): { referencedKeys: Set<string>, missingKeys: string[] } {
    const referencedKeys = new Set([...prompt.matchAll(REFERENCE_REGEX)].map(match => match[1]));
    const missingKeys: string[] = [];
    
    referencedKeys.forEach(key => {
        if (!references.has(key)) {
            missingKeys.push(key);
        }
    });

    return { referencedKeys, missingKeys };
}

/**
 * Constructs the array of Parts for the Gemini API request.
 */
function buildApiParts(prompt: string, image: typeof stagedImage | null, referencedKeys: Set<string>): Part[] {
    const parts: Part[] = [];
    const effectivePrompt = prompt || (image ? 'Describe this image.' : '');

    if (effectivePrompt) {
        parts.push({ text: effectivePrompt });
    }

    if (image) {
        // If there's a mask, the API expects prompt, original image, then mask image
        if (image.maskData) {
            parts.push({
                inlineData: {
                    data: image.data.split(',')[1],
                    mimeType: image.mimeType,
                },
            });
            parts.push({
                inlineData: {
                    data: image.maskData.split(',')[1],
                    mimeType: image.mimeType, // e.g., 'image/png'
                },
            });
        } else {
            // No mask, just send the original image
            parts.push({
                inlineData: {
                    data: image.data.split(',')[1],
                    mimeType: image.mimeType,
                },
            });
        }
    }
    
    referencedKeys.forEach(key => {
        const ref = references.get(key)!;
        parts.push({
            inlineData: {
                data: ref.data.split(',')[1],
                mimeType: ref.mimeType,
            },
        });
    });

    return parts;
}

/**
 * Processes the response from the Gemini API and adds messages to the chat.
 */
function processApiResponse(response: GenerateContentResponse) {
    let responseText = '';
    const responseImageDataUrls: string[] = [];

    for (const part of response.candidates[0].content.parts) {
        if (part.text) {
            responseText += part.text;
        } else if (part.inlineData) {
            responseImageDataUrls.push(`data:${part.inlineData.mimeType};base64,${part.inlineData.data}`);
        }
    }
    
    const fallbackText = responseImageDataUrls.length > 0 
        ? 'Here is the generated image.' 
        : 'I couldn\'t generate a response.';
        
    addMessage('ai', responseText || fallbackText, responseImageDataUrls);
}

// --- UI & MESSAGE HANDLING ---

/**
 * Adds a message to the chat history UI.
 */
function addMessage(sender: 'user' | 'ai', text: string, imageDataUrls?: string[] | null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    if (text) {
        const textNode = document.createElement('div');
        textNode.innerHTML = text; // Safe as only our own controlled HTML is used
        contentDiv.appendChild(textNode);
    }

    if (imageDataUrls && imageDataUrls.length > 0) {
        const imageGrid = document.createElement('div');
        imageGrid.className = 'message-image-grid';

        imageDataUrls.forEach((url, index) => {
            const imageContainer = document.createElement('div');
            imageContainer.className = 'image-container';

            const img = new Image();
            img.src = url;
            img.alt = text || `${sender} image ${index + 1}`;
            imageContainer.appendChild(img);

            if (sender === 'ai') {
                const actionsContainer = createAiImageActions(url, index);
                imageContainer.appendChild(actionsContainer);
            }
            
            imageGrid.appendChild(imageContainer);
        });
        contentDiv.appendChild(imageGrid);
    }
    
    messageDiv.appendChild(contentDiv);
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

/**
 * Creates the action buttons (edit, save, download) for an AI-generated image.
 */
function createAiImageActions(url: string, index: number): HTMLElement {
    const actionsContainer = document.createElement('div');
    actionsContainer.className = 'ai-image-actions';
    
    const editBtn = document.createElement('button');
    editBtn.className = 'image-action-btn';
    editBtn.innerHTML = '🎨';
    editBtn.title = 'Edit this Image';
    editBtn.setAttribute('aria-label', 'Edit this Image');
    editBtn.addEventListener('click', () => stageGeneratedImageForEditing(url));

    const saveBtn = document.createElement('button');
    saveBtn.className = 'image-action-btn';
    saveBtn.innerHTML = '💾';
    saveBtn.title = 'Save as Reference';
    saveBtn.setAttribute('aria-label', 'Save as Reference');
    saveBtn.addEventListener('click', () => {
        const mimeType = url.substring(url.indexOf(':') + 1, url.indexOf(';'));
        openSaveKeyModal(url, mimeType, 'generated');
    });

    const downloadBtn = document.createElement('a');
    downloadBtn.href = url;
    const mimeType = url.substring(url.indexOf(':') + 1, url.indexOf(';'));
    const extension = mimeType.split('/')[1] || 'png';
    downloadBtn.download = `gemini-image-${Date.now()}-${index + 1}.${extension}`;
    downloadBtn.className = 'image-action-btn';
    downloadBtn.innerHTML = '&#x2B07;';
    downloadBtn.title = 'Download Image';
    downloadBtn.setAttribute('aria-label', 'Download Image');
    
    actionsContainer.appendChild(editBtn);
    actionsContainer.appendChild(saveBtn);
    actionsContainer.appendChild(downloadBtn);
    return actionsContainer;
}

/**
 * Toggles the visibility of the loading indicator and disables/enables form controls.
 */
function toggleLoading(isLoading: boolean) {
    loadingIndicator.classList.toggle('hidden', !isLoading);
    sendBtn.disabled = isLoading;
    uploadBtn.disabled = isLoading;
    promptInput.disabled = isLoading;
}

/**
 * Resets the input area after a message is sent.
 */
function resetInputArea() {
    promptInput.value = '';
    promptInput.style.height = 'auto';
    imageUpload.value = '';
    stagedImage = null;
    imagePreviewContainer.classList.add('hidden');
    referencePickerPopover.classList.add('hidden');
}

/**
 * Handles file selection, reads the file as base64, and displays a preview.
 */
function handleFileSelect(event: Event) {
    const target = event.target as HTMLInputElement;
    const file = target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        const data = e.target?.result as string;
        stagedImage = { file, data, mimeType: file.type };
        imagePreview.src = data;
        imagePreviewContainer.classList.remove('hidden');
        openSaveKeyModal(data, file.type, 'upload');
    };
    reader.readAsDataURL(file);
}

/**
 * Sets a generated image as the staged image for further editing.
 */
function stageGeneratedImageForEditing(dataUrl: string) {
    const mimeType = dataUrl.substring(dataUrl.indexOf(':') + 1, dataUrl.indexOf(';'));
    const extension = mimeType.split('/')[1] || 'png';
    const filename = `generated-edit-${Date.now()}.${extension}`;
    const file = dataURLtoFile(dataUrl, filename);

    stagedImage = { file, data: dataUrl, mimeType };
    imagePreview.src = dataUrl;
    imagePreviewContainer.classList.remove('hidden');
    promptInput.scrollIntoView({ behavior: 'smooth', block: 'end' });
    promptInput.focus();
}

/**
 * Converts a base64 data URL string into a File object.
 */
function dataURLtoFile(dataurl: string, filename: string): File {
    const arr = dataurl.split(',');
    const mimeMatch = arr[0].match(/:(.*?);/);
    if (!mimeMatch) {
        throw new Error('Invalid data URL format');
    }
    const mime = mimeMatch[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, { type: mime });
}


// --- REFERENCE PICKER ---

function toggleReferencePicker() {
    const isHidden = referencePickerPopover.classList.toggle('hidden');
    if (!isHidden) {
        populateReferencePicker();
        referenceSearchInput.focus();
    }
}

function populateReferencePicker(filter = '') {
    referencePickerGrid.innerHTML = '';
    const lowerCaseFilter = filter.toLowerCase();

    for (const [key, { data }] of references.entries()) {
        if (key.toLowerCase().includes(lowerCaseFilter)) {
            const item = document.createElement('div');
            item.className = 'picker-item';
            item.title = key;
            item.innerHTML = `
                <img src="${data}" alt="${key}">
                <span>${key}</span>
            `;
            item.addEventListener('click', () => {
                insertReferenceAtCursor(key);
                toggleReferencePicker();
            });
            referencePickerGrid.appendChild(item);
        }
    }
}

function insertReferenceAtCursor(key: string) {
    const start = promptInput.selectionStart;
    const end = promptInput.selectionEnd;
    const text = promptInput.value;
    const newText = `${text.substring(0, start)}@${key} ${text.substring(end)}`;
    
    promptInput.value = newText;
    promptInput.focus();
    promptInput.selectionStart = promptInput.selectionEnd = start + key.length + 2;
}


// --- MODAL LOGIC ---

function openSaveKeyModal(imageData: string, mimeType: string, source: 'upload' | 'generated') {
    imageToSave = { data: imageData, mimeType };
    saveModalSource = source;
    
    saveKeyPreview.src = imageData;
    referenceKeyInput.value = '';
    keyErrorMessage.textContent = '';
    
    const isUpload = source === 'upload';
    useOnceBtn.classList.toggle('hidden', !isUpload);
    saveKeyBtn.textContent = isUpload ? 'Save & Use' : 'Save Reference';
    
    saveKeyModal.classList.remove('hidden');
    referenceKeyInput.focus();
}

function closeSaveKeyModal() {
    saveKeyModal.classList.add('hidden');
    imageToSave = null;
    saveModalSource = null;
}

function handleSaveKey() {
    if (!imageToSave) return;

    const key = referenceKeyInput.value.trim();
    keyErrorMessage.textContent = '';

    if (!key) {
        if (saveModalSource === 'upload') {
            closeSaveKeyModal(); // Treat as "Use Once"
            return;
        }
        keyErrorMessage.textContent = 'A key is required to save a reference.';
        return;
    }

    if (!KEY_REGEX.test(key)) {
        keyErrorMessage.textContent = 'Invalid key. Use only letters, numbers, and underscores.';
        return;
    }

    if (references.has(key)) {
        keyErrorMessage.textContent = 'This key is already in use. Please choose another.';
        return;
    }
    
    references.set(key, { data: imageToSave.data, mimeType: imageToSave.mimeType });
    addMessage('ai', `Image saved as reference with key: "<strong>${key}</strong>".`);
    closeSaveKeyModal();
}

function handleCancelSave() {
    if (saveModalSource === 'upload') {
        resetInputArea();
    }
    closeSaveKeyModal();
}

function openReferencesModal() {
    populateReferencesModal();
    referencesModal.classList.remove('hidden');
}

function closeReferencesModal() {
    referencesModal.classList.add('hidden');
}

function populateReferencesModal() {
    referencesList.innerHTML = '';
    
    if (references.size === 0) {
        referencesList.innerHTML = `<p class="empty-references-message">No image references saved yet.</p>`;
        return;
    }

    for (const [key, { data }] of references.entries()) {
        const item = createReferenceItemElement(key, data);
        referencesList.appendChild(item);
    }
}

function createReferenceItemElement(key: string, data: string): HTMLElement {
    const item = document.createElement('div');
    item.className = 'reference-item';
    
    item.innerHTML = `
        <img src="${data}" alt="${key}" class="reference-item__preview">
        <div class="reference-item__details">
            <input type="text" value="${key}" class="reference-item__input" data-old-key="${key}" aria-label="Reference key">
            <div class="reference-item__actions">
                <button class="reference-item__action-btn reference-item__use-btn" aria-label="Use this reference for editing" title="Use for editing">✏️</button>
                <button class="reference-item__action-btn reference-item__delete-btn" aria-label="Delete reference">&times;</button>
            </div>
        </div>
    `;

    const input = item.querySelector('.reference-item__input') as HTMLInputElement;
    const deleteBtn = item.querySelector('.reference-item__delete-btn')!;
    const useBtn = item.querySelector('.reference-item__use-btn')!;

    useBtn.addEventListener('click', () => {
        const ref = references.get(key);
        if (!ref) return;
        closeReferencesModal();
        stageGeneratedImageForEditing(ref.data);
    });

    deleteBtn.addEventListener('click', () => {
        if (confirm(`Are you sure you want to delete the reference "${key}"?`)) {
            references.delete(key);
            populateReferencesModal();
        }
    });

    const handleRename = () => {
        const oldKey = input.dataset.oldKey!;
        const newKey = input.value.trim();
        input.classList.remove('invalid');

        if (newKey === oldKey) return;

        if (!KEY_REGEX.test(newKey) || (references.has(newKey) && newKey !== oldKey)) {
            input.classList.add('invalid');
            input.value = oldKey;
            setTimeout(() => input.classList.remove('invalid'), 2000);
            return;
        }

        const refData = references.get(oldKey)!;
        references.delete(oldKey);
        references.set(newKey, refData);
        populateReferencesModal();
    };

    input.addEventListener('blur', handleRename);
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') input.blur();
        else if (e.key === 'Escape') {
            input.value = input.dataset.oldKey!;
            input.blur();
        }
    });

    return item;
}

// --- MASK EDITOR LOGIC ---

function openEditor() {
    if (!stagedImage) return;
    originalImage = new Image();
    originalImage.onload = () => {
        editorCanvas.width = originalImage!.naturalWidth;
        editorCanvas.height = originalImage!.naturalHeight;
        ctx.drawImage(originalImage!, 0, 0);
        editorModal.classList.remove('hidden');
        // Reset history and save initial state
        canvasHistory = [];
        saveCanvasState();
    };
    originalImage.src = stagedImage.data;
}

function closeEditor() {
    editorModal.classList.add('hidden');
    originalImage = null;
    canvasHistory = [];
    ctx.clearRect(0, 0, editorCanvas.width, editorCanvas.height);
}

function saveMask() {
    if (!stagedImage || canvasHistory.length === 0) return;

    const originalImageData = canvasHistory[0];
    const currentImageData = ctx.getImageData(0, 0, editorCanvas.width, editorCanvas.height);
    
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = editorCanvas.width;
    maskCanvas.height = editorCanvas.height;
    const maskCtx = maskCanvas.getContext('2d')!;
    const maskImageData = maskCtx.createImageData(editorCanvas.width, editorCanvas.height);

    const originalPixels = originalImageData.data;
    const currentPixels = currentImageData.data;
    const maskPixels = maskImageData.data;

    // Iterate through pixels. If a pixel has changed, make it white in the mask. Otherwise, black.
    for (let i = 0; i < originalPixels.length; i += 4) {
        const hasChanged = originalPixels[i] !== currentPixels[i] ||
                           originalPixels[i + 1] !== currentPixels[i + 1] ||
                           originalPixels[i + 2] !== currentPixels[i + 2] ||
                           originalPixels[i + 3] !== currentPixels[i + 3];
        
        if (hasChanged) {
            // White for masked area
            maskPixels[i] = 255;
            maskPixels[i + 1] = 255;
            maskPixels[i + 2] = 255;
            maskPixels[i + 3] = 255;
        } else {
            // Black for non-masked area
            maskPixels[i] = 0;
            maskPixels[i + 1] = 0;
            maskPixels[i + 2] = 0;
            maskPixels[i + 3] = 255;
        }
    }
    
    maskCtx.putImageData(maskImageData, 0, 0);
    stagedImage.maskData = maskCanvas.toDataURL(stagedImage.mimeType);

    // For the UI preview, show the image with the user's colorful drawings
    stagedImage.maskPreviewData = editorCanvas.toDataURL(stagedImage.mimeType);
    imagePreview.src = stagedImage.maskPreviewData;

    closeEditor();
}

function saveCanvasState() {
    canvasHistory.push(ctx.getImageData(0, 0, editorCanvas.width, editorCanvas.height));
}

function handleUndo() {
    if (canvasHistory.length > 1) { // Keep the initial state
        canvasHistory.pop();
        ctx.putImageData(canvasHistory[canvasHistory.length - 1], 0, 0);
    }
}

function getMousePos(e: MouseEvent): { x: number, y: number } {
    const rect = editorCanvas.getBoundingClientRect();
    const scaleX = editorCanvas.width / rect.width;
    const scaleY = editorCanvas.height / rect.height;
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY,
    };
}

function startDraw(e: MouseEvent) {
    isDrawing = true;
    const { x, y } = getMousePos(e);

    if (currentTool === 'brush' || currentTool === 'eraser') {
        ctx.beginPath();
        ctx.moveTo(x, y);
    } else if (currentTool === 'rectangle' || currentTool === 'circle') {
        shapeStartX = x;
        shapeStartY = y;
    }
}

function stopDraw(e: MouseEvent) {
    if (!isDrawing) return;
    isDrawing = false;
    
    // For shapes, draw the final shape on mouseup
    if (currentTool === 'rectangle' || currentTool === 'circle') {
        ctx.putImageData(canvasHistory[canvasHistory.length - 1], 0, 0); // Clear preview
        const { x, y } = getMousePos(e);
        drawShape(shapeStartX, shapeStartY, x, y);
    }
    
    saveCanvasState();
}

function draw(e: MouseEvent) {
    if (!isDrawing) return;
    const { x, y } = getMousePos(e);

    // Set common drawing properties
    ctx.lineWidth = parseInt(brushSize.value, 10);
    ctx.lineCap = 'round';
    const activeColor = (colorPalette.querySelector('.selected') as HTMLElement)?.dataset.color || '#E53935';
    ctx.strokeStyle = activeColor;
    ctx.fillStyle = activeColor;
    
    if (currentTool === 'brush') {
        ctx.globalCompositeOperation = 'source-over';
        ctx.lineTo(x, y);
        ctx.stroke();
    } else if (currentTool === 'eraser') {
        ctx.globalCompositeOperation = 'destination-out';
        ctx.lineTo(x, y);
        ctx.stroke();
    } else if (currentTool === 'rectangle' || currentTool === 'circle') {
        // Restore previous state for preview, then draw shape
        ctx.putImageData(canvasHistory[canvasHistory.length - 1], 0, 0);
        drawShape(shapeStartX, shapeStartY, x, y);
    }
}

function drawShape(x1: number, y1: number, x2: number, y2: number) {
    ctx.globalCompositeOperation = 'source-over';
    if (currentTool === 'rectangle') {
        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    } else if (currentTool === 'circle') {
        const radius = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
        ctx.beginPath();
        ctx.arc(x1, y1, radius, 0, 2 * Math.PI);
        ctx.fill();
    }
}

// --- HELP & TUTORIALS ---

function showInitialTutorial() {
    const tutorialText = `
        <h3>Welcome to the Ultimate AI Image Editor! ✨</h3>
        <p>Let's walk through how to create amazing images. Here’s a quick tutorial:</p>
        <ol style="margin-left: 1.5rem; padding-left: 0; line-height: 1.6;">
            <li style="margin-bottom: 0.75rem;">
                <strong>Upload an Image:</strong>
                Start by clicking the 📎 paperclip icon to upload your first image.
            </li>
            <li style="margin-bottom: 0.75rem;">
                <strong>Describe Your Edit:</strong>
                Once uploaded, simply type what you want to change in the text box. Try something like <em>"make the background a futuristic city"</em> or <em>"add a cute robot cat."</em>
            </li>
            <li style="margin-bottom: 0.75rem;">
                <strong>Save & Reuse Images:</strong>
                When you upload, you'll be asked for a "key". Giving it a key (e.g., "my_dog") saves it as a reference. You can then use it in other prompts, like <em>"place @my_dog in this new scene."</em> Use the @ button to pick from your saved images.
            </li>
            <li style="margin-bottom: 0.75rem;">
                <strong>Make Precise Changes:</strong>
                Need to edit a specific part of your image? Click the 🎨 paintbrush icon on the image preview to draw a "mask" over the area you want to change.
            </li>
        </ol>
        <p>You can manage all your saved references using the 📚 book icon in the header. For a full command list anytime, just type <strong>help</strong>.</p>
        <p>Ready to start creating? Upload an image!</p>
    `;
    addMessage('ai', tutorialText);
}

function showHelpMessage() {
    const helpText = `
        <h3>Welcome to the Ultimate AI Image Editor!</h3>
        <p>Here's a summary of what you can do:</p>
        <ul>
            <li><strong>Upload & Edit:</strong> Click the 📎 icon to upload an image. Then, type a prompt describing the changes you want (e.g., "make the sky blue").</li>
            <li><strong>Saving & Using Images:</strong> When you upload an image, a dialog will appear:
                <ul style="margin-top: 0.5rem; margin-bottom: 0.5rem;">
                    <li style="margin-top: 5px;">To <strong>save the image as a reference</strong> for later, enter a unique key (e.g., "cat_photo") and click "Save & Use".</li>
                    <li style="margin-top: 5px;">To <strong>use the image just once</strong> without saving, simply click "Use Once" (or leave the key blank and click "Save & Use").</li>
                </ul>
            </li>
            <li><strong>Use Saved References:</strong> Click the @ button to open a picker, or type <code>@your_key</code> in your prompt. For example: "add the @cat_photo into this scene."</li>
            <li><strong>Manage References:</strong> Click the 📚 icon in the header to view, rename, and delete all your saved image references.</li>
            <li><strong>Masking Tool (Precise Edits):</strong> After uploading an image, click the 🎨 icon on its preview. This opens an editor where you can draw to specify exactly where you want your edits to apply.</li>
            <li><strong>Download & Save Images:</strong> Hover over any image generated by the AI and click the ⬇️ icon to download it, or the 💾 icon to save it as a new reference.</li>
            <li><strong>Get Help:</strong> Type <code>help</code> at any time to see this summary again.</li>
        </ul>
    `;
    addMessage('ai', helpText);
}

// --- INITIALIZATION ---

function initializeApp() {
    // Form & Input Listeners
    chatForm.addEventListener('submit', handleFormSubmit);
    uploadBtn.addEventListener('click', () => imageUpload.click());
    imageUpload.addEventListener('change', handleFileSelect);
    removeImageBtn.addEventListener('click', resetInputArea);
    promptInput.addEventListener('input', () => {
        promptInput.style.height = 'auto';
        promptInput.style.height = `${promptInput.scrollHeight}px`;
    });

    // Reference Picker Listeners
    refPickerBtn.addEventListener('click', toggleReferencePicker);
    referenceSearchInput.addEventListener('input', () => populateReferencePicker(referenceSearchInput.value));


    // Editor Modal Listeners
    editMaskBtn.addEventListener('click', openEditor);
    editorCloseBtn.addEventListener('click', closeEditor);
    saveMaskBtn.addEventListener('click', saveMask);
    editorCanvas.addEventListener('mousedown', startDraw);
    editorCanvas.addEventListener('mouseup', stopDraw as (e: Event) => void);
    editorCanvas.addEventListener('mouseout', stopDraw as (e: Event) => void);
    editorCanvas.addEventListener('mousemove', draw);
    undoBtn.addEventListener('click', handleUndo);

    toolSelection.addEventListener('click', (e) => {
        const target = e.target as HTMLElement;
        // Fix: Cast the result of `closest` to HTMLElement to access `dataset`.
        const toolBtn = target.closest<HTMLElement>('.tool-btn');
        if (toolBtn?.dataset.tool) {
            toolSelection.querySelector('.active')?.classList.remove('active');
            toolBtn.classList.add('active');
            currentTool = toolBtn.dataset.tool;
        }
    });

    colorPalette.addEventListener('click', (e) => {
        const target = e.target as HTMLElement;
        // Fix: Cast the result of `closest` to HTMLElement to access `dataset`.
        const colorBtn = target.closest<HTMLElement>('.color-btn');
        if (colorBtn?.dataset.color) {
            colorPalette.querySelector('.selected')?.classList.remove('selected');
            colorBtn.classList.add('selected');
            // If user picks a color, switch to brush tool
            toolSelection.querySelector('.active')?.classList.remove('active');
            toolSelection.querySelector<HTMLButtonElement>('[data-tool="brush"]')?.classList.add('active');
            currentTool = 'brush';
        }
    });

    // Save Key Modal Listeners
    saveKeyBtn.addEventListener('click', handleSaveKey);
    referenceKeyInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') handleSaveKey();
    });
    useOnceBtn.addEventListener('click', closeSaveKeyModal);
    saveKeyModalCloseBtn.addEventListener('click', handleCancelSave);

    // References Management Modal Listeners
    manageReferencesBtn.addEventListener('click', openReferencesModal);
    referencesModalCloseBtn.addEventListener('click', closeReferencesModal);

    // Global Listeners
    window.addEventListener('click', (e) => {
      const target = e.target as HTMLElement;
      if (target === editorModal) closeEditor();
      if (target === saveKeyModal) handleCancelSave();
      if (target === referencesModal) closeReferencesModal();
      
      if (!referencePickerPopover.classList.contains('hidden')) {
        if (!referencePickerPopover.contains(target) && target !== refPickerBtn) {
            toggleReferencePicker();
        }
      }
    });

    // Show initial message
    showInitialTutorial();
}

initializeApp();