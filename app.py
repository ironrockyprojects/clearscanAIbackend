"""
ClearScan AI - Flask Backend
Dual-Engine Enhancement: SRCNN (PyTorch) → EDSR (OpenCV DNN)

Pipeline:
  RAW X-RAY
    → Stage 1: CLAHE + NLM Denoise          (OpenCV)
    → Stage 2: SRCNN 2x Super Resolution    (PyTorch)
    → Stage 3: EDSR  4x Super Resolution    (OpenCV DNN)
    → Stage 4: Sharpen + Edge + CLAHE + Gamma
  OUTPUT: High-quality enhanced X-ray

Install:
    pip install flask flask-cors opencv-contrib-python numpy requests supabase torch torchvision python-dotenv
    
Download EDSR model (place in backend/ folder):
    curl -L -o EDSR_x4.pb "https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/EDSR_x4.pb"
"""

import os
import uuid
import logging
from datetime import datetime, timezone

import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from dotenv import load_dotenv

import urllib.request

# ── Auto-download EDSR model on Railway server start ──────────
EDSR_AUTO_URL = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"

if not os.path.exists("EDSR_x4.pb"):
    logger.info("⬇️  EDSR model not found — downloading (~23MB)...")
    try:
        urllib.request.urlretrieve(EDSR_AUTO_URL, "EDSR_x4.pb")
        logger.info("✅ EDSR_x4.pb downloaded successfully")
    except Exception as e:
        logger.warning(f"⚠️  EDSR download failed: {e} — will use Lanczos fallback")

load_dotenv()

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Supabase ──────────────────────────────────────────────────
SUPABASE_URL         = os.environ.get("SUPABASE_URL", "YOUR_SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "YOUR_SERVICE_ROLE_KEY")
supabase: Client     = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
BUCKET_NAME          = "xrays"


# ══════════════════════════════════════════════════════════════
#  ENGINE 1 — SRCNN (PyTorch) — 2x Super Resolution
# ══════════════════════════════════════════════════════════════

try:
    import torch
    import torch.nn as nn

    class SRCNN(nn.Module):
        """
        Super-Resolution CNN (SRCNN)
        Architecture : 9-1-5 convolutional layers
        Input        : bicubic-upscaled grayscale float32 tensor [0,1]
        Output       : refined SR grayscale float32 tensor [0,1]
        """
        def __init__(self):
            super(SRCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
            self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
            self.conv3 = nn.Conv2d(32,  1, kernel_size=5, padding=2)
            self.relu  = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.conv3(x)
            return x

    srcnn_model = SRCNN()
    MODEL_PATH  = "srcnn_weights.pth"

    if os.path.exists(MODEL_PATH):
        srcnn_model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu")
        )
        SRCNN_TRAINED = True
        logger.info("✅ SRCNN — pre-trained weights loaded")
    else:
        SRCNN_TRAINED = False
        logger.warning("⚠️  SRCNN — no weights found, using guided-bicubic blend mode")

    srcnn_model.eval()
    TORCH_AVAILABLE = True

except ImportError:
    logger.warning("⚠️  PyTorch not installed — SRCNN engine skipped")
    TORCH_AVAILABLE = False
    SRCNN_TRAINED   = False
    srcnn_model     = None


# ══════════════════════════════════════════════════════════════
#  ENGINE 2 — EDSR (OpenCV DNN) — 4x Super Resolution
# ══════════════════════════════════════════════════════════════

EDSR_MODEL_PATH = "EDSR_x4.pb"
edsr_sr         = None
EDSR_AVAILABLE  = False

try:
    edsr_sr = cv2.dnn_superres.DnnSuperResImpl_create()
    edsr_sr.readModel(EDSR_MODEL_PATH)
    edsr_sr.setModel("edsr", 4)
    EDSR_AVAILABLE = True
    logger.info("✅ EDSR — 4x model loaded successfully")
except Exception as e:
    logger.warning(f"⚠️  EDSR — model not loaded: {e}")
    logger.warning("   Download: curl -L -o EDSR_x4.pb https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/EDSR_x4.pb")
    edsr_sr = None


# ══════════════════════════════════════════════════════════════
#  STAGE 1 — OpenCV Preprocessing
# ══════════════════════════════════════════════════════════════

def stage1_preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert to grayscale + CLAHE contrast boost + NLM denoising.
    Returns: uint8 grayscale image.
    """
    logger.info("  [Stage 1] CLAHE + NLM Denoise")

    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) \
           if len(img_bgr.shape) == 3 else img_bgr.copy()

    # CLAHE — contrast limited adaptive histogram equalization
    # Best for X-ray contrast enhancement
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    # Non-Local Means Denoising — removes sensor/compression noise
    denoised = cv2.fastNlMeansDenoising(
        contrast,
        h=8,
        templateWindowSize=7,
        searchWindowSize=21
    )

    logger.info(f"     Output: {denoised.shape[1]}x{denoised.shape[0]}")
    return denoised


# ══════════════════════════════════════════════════════════════
#  STAGE 2 — SRCNN 2x Super Resolution
# ══════════════════════════════════════════════════════════════

def stage2_srcnn(gray: np.ndarray) -> np.ndarray:
    """
    SRCNN 2x super resolution pass.
    Input is capped at 512px max dimension before upscaling
    so EDSR Stage 3 receives a manageable size (max 1024px).

    - Trained weights  → full SRCNN inference (2x)
    - No weights       → bicubic 2x + 15% SRCNN texture blend
    - No PyTorch       → pure bicubic 2x

    Returns: uint8 grayscale (2x input size, max 1024px).
    """
    logger.info("  [Stage 2] SRCNN — 2x Super Resolution")

    h, w = gray.shape

    # ── Cap input size to prevent EDSR timeout ──
    # Max input to SRCNN = 512px → max output = 1024px → EDSR gets 1024px max
    MAX_INPUT = 512
    if max(h, w) > MAX_INPUT:
        scale  = MAX_INPUT / max(h, w)
        new_w  = int(w * scale)
        new_h  = int(h * scale)
        gray   = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w   = gray.shape
        logger.info(f"     Capped input → {w}x{h}")

    # Bicubic upscale (SRCNN standard pre-processing step)
    bicubic_2x = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    if TORCH_AVAILABLE and srcnn_model is not None:
        import torch

        # Normalize to [0, 1] float32
        normalized = bicubic_2x.astype(np.float32) / 255.0

        # Build tensor [Batch=1, Channel=1, H, W]
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = srcnn_model(tensor)

        result_f = np.clip(output.squeeze().numpy(), 0.0, 1.0)
        result   = (result_f * 255).astype(np.uint8)

        if SRCNN_TRAINED:
            # Fully trained weights → use SRCNN output directly
            logger.info(f"     SRCNN trained inference → {result.shape[1]}x{result.shape[0]}")
            return result
        else:
            # Untrained → safe blend: 85% bicubic + 15% SRCNN texture hints
            # Prevents pure-black output from untrained weights
            blended = cv2.addWeighted(bicubic_2x, 0.85, result, 0.15, 0)
            logger.info(f"     SRCNN guided-bicubic blend → {blended.shape[1]}x{blended.shape[0]}")
            return blended
    else:
        logger.info(f"     Bicubic 2x fallback → {bicubic_2x.shape[1]}x{bicubic_2x.shape[0]}")
        return bicubic_2x


# ══════════════════════════════════════════════════════════════
#  STAGE 3 — EDSR 4x Super Resolution
# ══════════════════════════════════════════════════════════════

def stage3_edsr(gray: np.ndarray) -> np.ndarray:
    """
    EDSR (Enhanced Deep Super Resolution) 4x upscaling.
    Input is capped at 256px max dimension for CPU performance.
    256px input → 1024px output — plenty of clinical detail.

    - EDSR model available → full AI 4x super resolution
    - Fallback             → Lanczos 4x + quality sharpening

    Returns: uint8 grayscale (4x input size).
    """
    logger.info("  [Stage 3] EDSR — 4x Super Resolution")

    if EDSR_AVAILABLE and edsr_sr is not None:
        h, w = gray.shape

        # ── Cap input for EDSR — critical for CPU performance ──
        # 256px → EDSR → 1024px output (high quality, fast on CPU)
        MAX_EDSR_INPUT = 256
        if max(h, w) > MAX_EDSR_INPUT:
            scale  = MAX_EDSR_INPUT / max(h, w)
            new_w  = int(w * scale)
            new_h  = int(h * scale)
            gray   = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"     Resized EDSR input → {new_w}x{new_h}")

        # EDSR requires BGR input
        bgr          = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        upscaled_bgr = edsr_sr.upsample(bgr)
        result       = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2GRAY)
        logger.info(f"     EDSR AI 4x output → {result.shape[1]}x{result.shape[0]}")

    else:
        # Lanczos is the best non-AI upscaling algorithm
        h, w   = gray.shape
        result = cv2.resize(
            gray,
            (w * 4, h * 4),
            interpolation=cv2.INTER_LANCZOS4
        )
        logger.info(f"     Lanczos 4x fallback → {result.shape[1]}x{result.shape[0]}")

    return result


# ══════════════════════════════════════════════════════════════
#  STAGE 4 — Post-Processing
# ══════════════════════════════════════════════════════════════

def stage4_postprocess(gray: np.ndarray) -> np.ndarray:
    """
    Final clinical-quality enhancement pass:
    1. Unsharp masking        → crisp edge sharpening
    2. Laplacian enhancement  → bone / tissue boundary detail
    3. Final CLAHE            → optimal display contrast
    4. Gamma correction       → balanced brightness output

    Returns: uint8 BGR image ready to JPEG encode.
    """
    logger.info("  [Stage 4] Sharpen + Edge + CLAHE + Gamma")

    # 1. Unsharp masking — strong sharpening pass
    gaussian  = cv2.GaussianBlur(gray, (0, 0), 2.5)
    sharpened = cv2.addWeighted(gray, 1.9, gaussian, -0.9, 0)

    # 2. Laplacian edge enhancement — highlights structural boundaries
    lap     = cv2.Laplacian(sharpened, cv2.CV_64F, ksize=3)
    lap_abs = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
    edged   = cv2.addWeighted(sharpened, 1.0, lap_abs, 0.25, 0)

    # 3. Final CLAHE pass for perfect display contrast
    clahe_final = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    final_gray  = clahe_final.apply(edged)

    # 4. Gamma correction — slightly brighten the output
    gamma = 1.15
    lut   = np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
        dtype=np.uint8
    )
    corrected = cv2.LUT(final_gray, lut)

    # Grayscale → BGR for JPEG encoding
    output_bgr = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
    logger.info(f"     Final output: {output_bgr.shape[1]}x{output_bgr.shape[0]}")
    return output_bgr


# ══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def enhance_xray(image_bytes: bytes) -> bytes:
    """
    Full dual-engine enhancement pipeline:

    RAW IMAGE
      ↓  Stage 1 : CLAHE contrast + NLM denoise         (OpenCV)
      ↓  Stage 2 : SRCNN 2x super resolution            (PyTorch) — capped at 512px input
      ↓  Stage 3 : EDSR  4x super resolution            (OpenCV DNN) — capped at 256px input
      ↓  Stage 4 : Unsharp + Laplacian + CLAHE + Gamma  (OpenCV)
    ENHANCED IMAGE — high quality, optimized for clinical review
    """
    logger.info("=" * 56)
    logger.info("🔬 ClearScan AI — Dual-Engine Pipeline START")
    logger.info("=" * 56)

    # Decode image bytes → numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image — ensure JPEG or PNG format.")

    logger.info(f"📥 Input  : {img.shape[1]}x{img.shape[0]}px  ({len(image_bytes)//1024} KB)")

    # Run all 4 stages
    s1 = stage1_preprocess(img)    # CLAHE + Denoise
    s2 = stage2_srcnn(s1)          # SRCNN 2x
    s3 = stage3_edsr(s2)           # EDSR 4x
    s4 = stage4_postprocess(s3)    # Sharpen + Edge + CLAHE + Gamma

    logger.info(f"📤 Output : {s4.shape[1]}x{s4.shape[0]}px")

    # Encode to high-quality JPEG
    ok, buf = cv2.imencode('.jpg', s4, [cv2.IMWRITE_JPEG_QUALITY, 97])
    if not ok:
        raise RuntimeError("JPEG encoding failed.")

    out = buf.tobytes()
    logger.info(f"✅ Done   : {len(out) // 1024} KB output")
    logger.info("=" * 56)
    return out


# ══════════════════════════════════════════════════════════════
#  Flask Routes
# ══════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    """Health check — returns engine status."""
    return jsonify({
        'status'   : 'ok',
        'engines'  : {
            'srcnn_pytorch': TORCH_AVAILABLE,
            'srcnn_trained': SRCNN_TRAINED,
            'edsr_opencv'  : EDSR_AVAILABLE,
        },
        'pipeline' : 'CLAHE+Denoise → SRCNN 2x (cap 512px) → EDSR 4x (cap 256px) → Sharpen+Edge+CLAHE+Gamma',
        'timestamp': datetime.now(timezone.utc).isoformat(),
    })


@app.route('/enhance', methods=['POST'])
def enhance():
    """
    POST /enhance
    Body   : { "image_url": "...", "user_id": "..." }
    Returns: { "success": true, "enhanced_url": "..." }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON body provided'}), 400

        image_url = data.get('image_url')
        user_id   = data.get('user_id', 'anonymous')

        if not image_url:
            return jsonify({'success': False, 'error': 'image_url is required'}), 400

        logger.info(f"📨 Enhancement request — user: {user_id}")

        # 1. Download original image from Supabase Storage
        logger.info("⬇️  Downloading original image...")
        resp = requests.get(image_url, timeout=60)
        resp.raise_for_status()
        logger.info(f"   Downloaded: {len(resp.content) // 1024} KB")

        # 2. Run dual-engine enhancement pipeline
        enhanced_bytes = enhance_xray(resp.content)

        # 3. Upload enhanced image back to Supabase Storage
        filename = f"{user_id}/enhanced_{uuid.uuid4().hex}.jpg"
        logger.info(f"⬆️  Uploading enhanced image: {filename}")
        supabase.storage.from_(BUCKET_NAME).upload(
            filename,
            enhanced_bytes,
            file_options={"content-type": "image/jpeg"}
        )

        # 4. Get and return public URL
        url = supabase.storage.from_(BUCKET_NAME).get_public_url(filename)
        logger.info(f"🎉 Enhanced image ready!")

        return jsonify({
            'success'      : True,
            'enhanced_url' : url,
            'filename'     : filename,
        })

    except requests.RequestException as e:
        logger.error(f"❌ Download error: {e}")
        return jsonify({'success': False, 'error': f'Image download failed: {str(e)}'}), 502

    except ValueError as e:
        logger.error(f"❌ Image decode error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 422

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    logger.info("🚀 ClearScan AI — Dual-Engine Server Starting")
    logger.info(f"   SRCNN PyTorch  : {'✅ Available'       if TORCH_AVAILABLE else '❌ Not installed'}")
    logger.info(f"   SRCNN Weights  : {'✅ Trained'         if SRCNN_TRAINED   else '⚠️  Bicubic blend mode'}")
    logger.info(f"   EDSR OpenCV    : {'✅ 4x AI loaded'    if EDSR_AVAILABLE  else '⚠️  Lanczos fallback'}")
    logger.info(f"   SRCNN size cap : 512px input → 1024px output")
    logger.info(f"   EDSR  size cap : 256px input → 1024px output")
    logger.info(f"   Pipeline       : CLAHE → SRCNN 2x → EDSR 4x → PostProcess")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)