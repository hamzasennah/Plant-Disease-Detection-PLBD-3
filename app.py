# -*- coding: utf-8 -*-
"""
AGRISCAN AI — Interface Gradio avec double confirmation + détection hors base
Utilise deux modèles entraînés : EfficientNet-B0 + ResNet-50.

NOUVEAU : si l'utilisateur soumet une image d'un végétal/fruit qui n'est PAS
dans les données d'entraînement (poivron / pomme de terre / tomate uniquement),
l'interface affiche un message clair "Plante non reconnue".

Détection hors base combinant 3 critères :
  1. Confiance max faible  (seuil : CONF_THRESHOLD = 0.55)
  2. Entropie élevée       (seuil : ENTROPY_THRESHOLD = 2.2)
  3. Désaccord des modèles (les deux prédisent des classes différentes)

Installation : !pip -q install gradio
"""

from pathlib import Path
import math

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image


# ============================================================
# 1. CONFIGURATION
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Plantes reconnues par le modèle ──────────────────────────────────────────
KNOWN_PLANTS = {"Poivron", "Pomme de terre", "Tomate"}

CLASSES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]
NUM_CLASSES = len(CLASSES)

# ── Seuils de détection hors-base ─────────────────────────────────────────────
#  Confiance : probabilité max de la moyenne des deux modèles.
#    < CONF_THRESHOLD  →  signal "hors base"
CONF_THRESHOLD = 0.55

#  Entropie de Shannon sur les 15 probabilités (max théorique ≈ log2(15) ≈ 3.91).
#    > ENTROPY_THRESHOLD  →  signal "hors base"
ENTROPY_THRESHOLD = 2.2

# Pour déclencher le message "hors base", on exige que
# AU MOINS 2 des 3 critères suivants soient vrais :
#   A. confiance < CONF_THRESHOLD
#   B. entropie  > ENTROPY_THRESHOLD
#   C. les deux modèles sont en désaccord
# Cela évite les faux positifs sur des images simplement difficiles.

# ── Chemins des checkpoints ──────────────────────────────────────────────────
EFFICIENTNET_CHECKPOINT = None
RESNET_CHECKPOINT       = None

EFFICIENTNET_CANDIDATES = [
    "/kaggle/working/efficientnet_best.pth",
    "/kaggle/working/efficientnet_model.pth",
    "/content/efficientnet_best.pth",
    "/content/efficientnet_model.pth",
    "efficientnet_best.pth",
    "efficientnet_model.pth",
]

RESNET_CANDIDATES = [
    "/kaggle/working/resnet50_best.pth",
    "/kaggle/working/resnet50_model.pth",
    "/kaggle/working/resnaet50_model.pth",
    "/content/resnet50_best.pth",
    "/content/resnet50_model.pth",
    "/content/resnaet50_model.pth",
    "resnet50_best.pth",
    "resnet50_model.pth",
    "resnaet50_model.pth",
]


# ============================================================
# 2. INFORMATIONS MALADIES
# ============================================================
DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "plant": "Poivron", "emoji": "🫑", "status": "Malade",
        "disease": "Tache bactérienne", "level": "high",
        "description": "Présence probable d'une maladie bactérienne provoquant des taches foncées sur les feuilles.",
        "advice": "Isoler la plante, retirer les feuilles très atteintes, éviter de mouiller le feuillage et demander conseil à un spécialiste agricole."
    },
    "Pepper__bell___healthy": {
        "plant": "Poivron", "emoji": "🫑", "status": "Sain",
        "disease": "Aucune maladie détectée", "level": "healthy",
        "description": "La feuille ressemble à une feuille saine selon le modèle.",
        "advice": "Continuer un arrosage régulier au pied, surveiller l'apparition de taches et garder une bonne aération."
    },
    "Potato___Early_blight": {
        "plant": "Pomme de terre", "emoji": "🥔", "status": "Malade",
        "disease": "Alternariose / mildiou précoce", "level": "medium",
        "description": "Taches brunes souvent circulaires, généralement visibles sur les feuilles plus âgées.",
        "advice": "Retirer les feuilles atteintes, éviter l'humidité excessive et appliquer un traitement autorisé après avis professionnel."
    },
    "Potato___Late_blight": {
        "plant": "Pomme de terre", "emoji": "🥔", "status": "Malade",
        "disease": "Mildiou tardif", "level": "critical",
        "description": "Maladie grave pouvant évoluer rapidement, surtout avec humidité et températures favorables.",
        "advice": "Agir rapidement : isoler les plants suspects, limiter l'humidité et demander confirmation à un technicien."
    },
    "Potato___healthy": {
        "plant": "Pomme de terre", "emoji": "🥔", "status": "Sain",
        "disease": "Aucune maladie détectée", "level": "healthy",
        "description": "La feuille est classée comme saine par le modèle.",
        "advice": "Maintenir de bonnes pratiques : arrosage au pied, observation régulière et bonne rotation des cultures."
    },
    "Tomato_Bacterial_spot": {
        "plant": "Tomate", "emoji": "🍅", "status": "Malade",
        "disease": "Tache bactérienne", "level": "high",
        "description": "Petites taches sombres pouvant apparaître sur les feuilles et parfois sur les fruits.",
        "advice": "Éviter l'arrosage sur les feuilles, retirer les parties touchées et désinfecter les outils."
    },
    "Tomato_Early_blight": {
        "plant": "Tomate", "emoji": "🍅", "status": "Malade",
        "disease": "Mildiou précoce / alternariose", "level": "medium",
        "description": "Taches brunes avec parfois un aspect en cercles concentriques.",
        "advice": "Supprimer les feuilles atteintes, pailler le sol pour éviter les éclaboussures."
    },
    "Tomato_Late_blight": {
        "plant": "Tomate", "emoji": "🍅", "status": "Malade",
        "disease": "Mildiou tardif", "level": "critical",
        "description": "Maladie sérieuse qui peut se propager rapidement en conditions humides.",
        "advice": "Isoler la plante, réduire l'humidité foliaire et confirmer le diagnostic avec un spécialiste."
    },
    "Tomato_Leaf_Mold": {
        "plant": "Tomate", "emoji": "🍅", "status": "Malade",
        "disease": "Moisissure foliaire", "level": "medium",
        "description": "Souvent observée en conditions humides, avec jaunissement et zones de moisissure.",
        "advice": "Améliorer la ventilation, réduire l'humidité et espacer les plants."
    },
    "Tomato_Septoria_leaf_spot": {
        "plant": "Tomate", "emoji": "🍅", "status": "Malade",
        "disease": "Septoriose", "level": "medium",
        "description": "Petites taches circulaires, souvent nombreuses, pouvant affaiblir progressivement la plante.",
        "advice": "Retirer les feuilles touchées et ne pas laisser de débris végétaux au sol."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "plant": "Tomate", "emoji": "🍅", "status": "Stress / ravageur",
        "disease": "Acariens", "level": "medium",
        "description": "Petits points jaunes et affaiblissement possible liés à des ravageurs minuscules.",
        "advice": "Observer le dessous des feuilles et demander conseil pour une méthode de lutte adaptée."
    },
    "Tomato__Target_Spot": {
        "plant": "Tomate", "emoji": "🍅", "status": "Malade",
        "disease": "Tache cible", "level": "medium",
        "description": "Taches arrondies pouvant rappeler une cible, parfois visibles sur feuilles et tiges.",
        "advice": "Améliorer l'aération et supprimer les feuilles très atteintes."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "plant": "Tomate", "emoji": "🍅", "status": "Malade",
        "disease": "Virus de l'enroulement jaune", "level": "critical",
        "description": "Feuilles jaunes et enroulées, croissance ralentie et baisse possible de production.",
        "advice": "Isoler les plants suspects et demander confirmation à un professionnel."
    },
    "Tomato__Tomato_mosaic_virus": {
        "plant": "Tomate", "emoji": "🍅", "status": "Malade",
        "disease": "Virus de la mosaïque", "level": "high",
        "description": "Aspect mosaïque vert/jaune et déformation possible des feuilles.",
        "advice": "Désinfecter les outils et confirmer le diagnostic."
    },
    "Tomato_healthy": {
        "plant": "Tomate", "emoji": "🍅", "status": "Sain",
        "disease": "Aucune maladie détectée", "level": "healthy",
        "description": "La feuille est classée comme saine par le modèle.",
        "advice": "Continuer la surveillance, arroser au pied et maintenir une bonne aération."
    },
}

LEVEL_STYLE = {
    "healthy":  {"label": "Sain",     "color": "#16a34a", "bg": "#dcfce7", "icon": "✅"},
    "low":      {"label": "Faible",   "color": "#65a30d", "bg": "#ecfccb", "icon": "🟢"},
    "medium":   {"label": "Modéré",   "color": "#d97706", "bg": "#fef3c7", "icon": "🟡"},
    "high":     {"label": "Élevé",    "color": "#ea580c", "bg": "#ffedd5", "icon": "🟠"},
    "critical": {"label": "Critique", "color": "#dc2626", "bg": "#fee2e2", "icon": "🔴"},
}


# ============================================================
# 3. CHARGEMENT DES MODÈLES
# ============================================================
def find_checkpoint(manual_path, candidates, keyword):
    if manual_path and Path(manual_path).exists():
        return str(manual_path)
    for path in candidates:
        if Path(path).exists():
            return str(path)
    for root in ["/kaggle/working", "/content", "."]:
        root_path = Path(root)
        if root_path.exists():
            matches = [p for p in root_path.rglob("*.pth") if keyword.lower() in p.name.lower()]
            if matches:
                return str(sorted(matches)[0])
    raise FileNotFoundError(
        f"Aucun checkpoint trouvé pour '{keyword}'. "
        "Vérifie le nom du fichier .pth ou modifie la configuration."
    )


def build_efficientnet():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model


def build_resnet():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def load_state(model, checkpoint_path):
    state = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def load_two_models():
    eff_path = find_checkpoint(EFFICIENTNET_CHECKPOINT, EFFICIENTNET_CANDIDATES, "efficient")
    res_path = find_checkpoint(RESNET_CHECKPOINT, RESNET_CANDIDATES, "res")
    efficientnet = load_state(build_efficientnet(), eff_path)
    resnet       = load_state(build_resnet(),       res_path)
    print(f"✅ EfficientNet : {eff_path}")
    print(f"✅ ResNet-50    : {res_path}")
    print(f"✅ Device        : {DEVICE}")
    return efficientnet, resnet


EFFICIENTNET_MODEL, RESNET_MODEL = load_two_models()


# ============================================================
# 4. TRANSFORMATION
# ============================================================
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ============================================================
# 5. DÉTECTION HORS BASE (Out-Of-Distribution)
# ============================================================
def shannon_entropy(probs: torch.Tensor) -> float:
    """
    Entropie de Shannon sur un vecteur de probabilités (base 2).
    Maximum théorique pour 15 classes : log2(15) ≈ 3.91.
    Une entropie élevée signifie que le modèle est incertain → image inconnue probable.
    """
    probs_np = probs.cpu().numpy()
    # éviter log(0)
    probs_np = probs_np + 1e-10
    probs_np = probs_np / probs_np.sum()
    entropy = -sum(p * math.log2(p) for p in probs_np)
    return entropy


def is_out_of_distribution(
    eff_probs: torch.Tensor,
    res_probs: torch.Tensor,
    eff_idx: int,
    res_idx: int,
) -> tuple[bool, dict]:
    """
    Retourne (True, détails) si l'image est probablement hors base de données.

    Critères (au moins 2/3 doivent être vrais) :
      A — Confiance faible  : max(avg_probs) < CONF_THRESHOLD
      B — Entropie élevée   : entropy(avg_probs) > ENTROPY_THRESHOLD
      C — Modèles désaccord : eff_idx != res_idx
    """
    avg_probs = (eff_probs + res_probs) / 2.0

    max_conf = float(avg_probs.max().item())
    entropy  = shannon_entropy(avg_probs)
    disagree = eff_idx != res_idx

    criterion_A = max_conf < CONF_THRESHOLD     # confiance insuffisante
    criterion_B = entropy  > ENTROPY_THRESHOLD  # trop d'incertitude
    criterion_C = disagree                      # les deux modèles divergent

    triggered = sum([criterion_A, criterion_B, criterion_C])
    ood = triggered >= 2   # on exige au moins 2 critères sur 3

    details = {
        "max_conf":    max_conf,
        "entropy":     entropy,
        "disagree":    disagree,
        "criterion_A": criterion_A,
        "criterion_B": criterion_B,
        "criterion_C": criterion_C,
        "triggered":   triggered,
    }
    return ood, details


# ============================================================
# 6. RENDU HTML
# ============================================================
EMPTY_HTML = """
<div class="empty-card">
  <div class="empty-icon">🌿</div>
  <h2>Prêt pour l'analyse</h2>
  <p>Importe une image nette d'une feuille de <b>poivron</b>, <b>pomme de terre</b>
     ou <b>tomate</b>, puis clique sur <b>Analyser</b>.</p>
  <p class="small-note">L'interface affiche l'accord : <b>2/2</b> ou <b>1/2</b>, sans pourcentage.</p>
</div>
"""


def render_ood_html(details: dict) -> str:
    """
    Affiche le message "Plante non reconnue" avec les raisons détaillées.
    """
    reasons = []
    if details["criterion_A"]:
        reasons.append(
            f"<li><b>Confiance insuffisante</b> — les deux modèles ne sont pas "
            f"assez certains de leur réponse (score : {details['max_conf']:.0%}).</li>"
        )
    if details["criterion_B"]:
        reasons.append(
            f"<li><b>Incertitude élevée</b> — les probabilités sont trop dispersées "
            f"sur les 15 classes (entropie : {details['entropy']:.2f} / 3.91 max).</li>"
        )
    if details["criterion_C"]:
        reasons.append(
            "<li><b>Désaccord entre les modèles</b> — EfficientNet-B0 et ResNet-50 "
            "ne donnent pas la même classification.</li>"
        )
    reasons_html = "\n".join(reasons)

    return f"""
    <div class="result-card">

      <!-- Bannière principale -->
      <div class="ood-banner">
        <div class="ood-icon">🚫</div>
        <div>
          <div class="ood-title">Plante non reconnue</div>
          <div class="ood-subtitle">
            Cette image ne correspond à aucune des plantes de notre base de données.
          </div>
        </div>
      </div>

      <!-- Plantes supportées -->
      <div class="supported-box">
        <h3>🌱 Plantes actuellement reconnues par AgriScan AI</h3>
        <div class="plant-pills">
          <span class="plant-pill">🫑 Poivron (bell pepper)</span>
          <span class="plant-pill">🥔 Pomme de terre</span>
          <span class="plant-pill">🍅 Tomate</span>
        </div>
        <p class="supported-note">
          Si vous avez soumis une image de <b>pomme, maïs, raisin, fraise, cerise,
          pêche, orange, citron</b> ou toute autre plante, notre système ne peut pas
          la diagnostiquer. Veuillez utiliser une image de l'une des trois plantes ci-dessus.
        </p>
      </div>

      <!-- Raisons techniques -->
      <div class="ood-reasons">
        <h3>🔍 Pourquoi cette image a-t-elle été rejetée ?</h3>
        <ul>{reasons_html}</ul>
        <p class="ood-criteria-note">
          ℹ️ La détection hors-base requiert au moins <b>2 signaux sur 3</b>
          (confiance faible · entropie élevée · désaccord des modèles).
          Ici : <b>{details['triggered']}/3</b> signaux détectés.
        </p>
      </div>

      <!-- Conseils -->
      <div class="ood-tips">
        <h3>💡 Que faire ?</h3>
        <ul>
          <li>Vérifiez que votre image montre bien une <b>feuille</b> (pas un fruit entier, pas du sol).</li>
          <li>Assurez-vous que la feuille est <b>nette</b>, bien éclairée et occupe la majorité du cadre.</li>
          <li>Essayez avec une autre photo de la <b>même plante</b>.</li>
          <li>Si votre plante est différente (maïs, vigne, etc.), notre modèle <b>ne peut pas</b> la diagnostiquer.</li>
        </ul>
      </div>

      <div class="disclaimer">
        AgriScan AI est entraîné uniquement sur des feuilles de poivron, pomme de terre et tomate.
        Tout autre type de végétal produira ce message de rejet.
      </div>
    </div>
    """


def model_card(model_name: str, pred_idx: int) -> str:
    key   = CLASSES[pred_idx]
    info  = DISEASE_INFO[key]
    style = LEVEL_STYLE[info["level"]]
    return f"""
    <div class="model-card">
      <div class="model-name">{model_name}</div>
      <div class="model-pred">
        <span class="plant-emoji">{info['emoji']}</span>
        <span>{info['plant']} — {info['disease']}</span>
      </div>
      <div class="mini-meta">
        <span style="background:{style['bg']}; color:{style['color']};
                     border-color:{style['color']}44;">
          {style['icon']} {style['label']}
        </span>
      </div>
    </div>
    """


def render_result_html(eff_idx: int, res_idx: int) -> str:
    """
    Affiche le résultat normal quand la plante est reconnue.
    """
    avg_idx = eff_idx if eff_idx == res_idx else eff_idx  # priorité à EffNet si désaccord
    # On utilise quand même la moyenne pour le diagnostic final
    final_key   = CLASSES[avg_idx]
    final_info  = DISEASE_INFO[final_key]
    final_style = LEVEL_STYLE[final_info["level"]]

    same = eff_idx == res_idx
    agreement_score = "2/2" if same else "1/2"
    if same:
        agr_title = "Diagnostic confirmé"
        agr_text  = "Les deux modèles donnent la même classification. Accord 2/2."
        agr_cls   = "agreement-good"
        agr_icon  = "✅"
    else:
        agr_title = "Diagnostic non confirmé"
        agr_text  = ("Les deux modèles donnent des classifications différentes. "
                     "Accord 1/2. Une vérification avec une autre image est recommandée.")
        agr_cls   = "agreement-warning"
        agr_icon  = "⚠️"

    model_cards = model_card("EfficientNet-B0", eff_idx) + model_card("ResNet-50", res_idx)

    return f"""
    <div class="result-card">
      <div class="score-big {agr_cls}">
        <div class="score-label">Accord des modèles</div>
        <div class="score-value">{agreement_score}</div>
      </div>

      <div class="agreement-box {agr_cls}">
        <div class="agreement-score">{agr_icon} {agr_title}</div>
        <p>{agr_text}</p>
      </div>

      <div class="result-hero"
           style="background: linear-gradient(135deg,{final_style['bg']} 0%,#ffffff 100%);
                  border-color:{final_style['color']}33;">
        <div class="hero-left">
          <div class="big-emoji">{final_info['emoji']}</div>
          <div>
            <div class="overline">Diagnostic final proposé</div>
            <h2>{final_info['plant']} — {final_info['disease']}</h2>
            <p>{final_info['status']}</p>
          </div>
        </div>
        <div class="severity"
             style="background:{final_style['bg']}; color:{final_style['color']};
                    border-color:{final_style['color']}55;">
          <span>{final_style['icon']}</span>
          <b>{final_style['label']}</b>
        </div>
      </div>

      <div class="models-grid">{model_cards}</div>

      <div class="info-grid">
        <div class="info-box">
          <h3>📌 Explication</h3>
          <p>{final_info['description']}</p>
        </div>
        <div class="info-box">
          <h3>🧭 Conseils</h3>
          <p>{final_info['advice']}</p>
        </div>
      </div>

      <div class="disclaimer">
        Ce résultat est une aide à la décision. Il ne remplace pas l'avis d'un agronome.
      </div>
    </div>
    """


# ============================================================
# 7. FONCTION PRINCIPALE DE PRÉDICTION
# ============================================================
def predict_leaf(image: Image.Image) -> str:
    if image is None:
        return EMPTY_HTML

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    # ── Inférence des deux modèles ───────────────────────────────────────────
    with torch.no_grad():
        eff_logits = EFFICIENTNET_MODEL(tensor)
        res_logits = RESNET_MODEL(tensor)

    eff_probs = F.softmax(eff_logits, dim=1).squeeze(0)
    res_probs = F.softmax(res_logits, dim=1).squeeze(0)

    eff_idx = int(torch.argmax(eff_probs).item())
    res_idx = int(torch.argmax(res_probs).item())

    # ── Détection hors base ──────────────────────────────────────────────────
    ood, details = is_out_of_distribution(eff_probs, res_probs, eff_idx, res_idx)

    if ood:
        return render_ood_html(details)

    # ── Résultat normal ──────────────────────────────────────────────────────
    return render_result_html(eff_idx, res_idx)


# ============================================================
# 8. CSS
# ============================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.gradio-container {
  max-width: 1280px !important;
  margin: auto !important;
  background: #f8fafc !important;
  font-family: 'Inter', sans-serif !important;
}
* { font-family: 'Inter', sans-serif !important; }

/* ── Header ── */
.main-header {
  background: radial-gradient(circle at top right, rgba(255,255,255,0.24), transparent 30%),
              linear-gradient(135deg, #064e3b 0%, #047857 45%, #10b981 100%);
  color: white;
  padding: 42px;
  border-radius: 28px;
  margin-bottom: 28px;
  box-shadow: 0 24px 70px rgba(5,150,105,0.28);
  position: relative;
  overflow: hidden;
}
.main-header:after {
  content: '🌱';
  position: absolute;
  right: 38px; bottom: -36px;
  font-size: 170px;
  opacity: 0.12;
}
.brand-pill {
  display: inline-flex;
  gap: 10px; align-items: center;
  background: rgba(255,255,255,0.16);
  border: 1px solid rgba(255,255,255,0.22);
  border-radius: 999px;
  padding: 8px 14px;
  font-size: 13px; font-weight: 700;
  letter-spacing: 0.4px;
  margin-bottom: 16px;
}
.main-header h1 {
  font-size: 44px !important; line-height: 1.05 !important;
  margin: 0 0 14px 0 !important; font-weight: 800 !important;
}
.main-header p { max-width: 760px; margin: 0 !important; font-size: 16px; line-height: 1.7; opacity: .94; }
.header-tags { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 22px; }
.header-tag {
  padding: 9px 14px; border-radius: 999px;
  background: rgba(255,255,255,0.14);
  border: 1px solid rgba(255,255,255,0.18);
  font-size: 13px; font-weight: 700;
}

/* ── Panels ── */
.panel {
  background: white; border: 1px solid #e5e7eb;
  border-radius: 24px; padding: 26px !important;
  box-shadow: 0 10px 35px rgba(15,23,42,0.06);
}
.section-title {
  color: #065f46; font-size: 13px !important;
  text-transform: uppercase; letter-spacing: 1px;
  font-weight: 800 !important; margin-bottom: 18px !important;
}
.upload-note {
  background: #ecfdf5; color: #064e3b;
  border: 1px solid #bbf7d0;
  padding: 16px; border-radius: 16px;
  font-size: 13.5px; line-height: 1.7; margin-top: 16px;
}
.predict-btn {
  background: linear-gradient(135deg,#059669,#047857) !important;
  color: white !important; border: 0 !important;
  border-radius: 16px !important; padding: 15px 22px !important;
  font-size: 16px !important; font-weight: 800 !important;
  box-shadow: 0 10px 24px rgba(5,150,105,0.28) !important;
}

/* ── Empty state ── */
.empty-card {
  text-align: center; padding: 72px 28px;
  background: linear-gradient(180deg,#ffffff,#f8fafc);
  border: 1px dashed #cbd5e1; border-radius: 22px; color: #475569;
}
.empty-icon { font-size: 70px; margin-bottom: 10px; }
.empty-card h2 { margin: 0 0 10px 0 !important; color: #0f172a; font-weight: 800 !important; }
.empty-card p  { margin: 8px 0 !important; line-height: 1.6; }
.small-note    { font-size: 12.5px; color: #64748b; }

/* ── OOD banner ── */
.ood-banner {
  display: flex; align-items: center; gap: 22px;
  background: linear-gradient(135deg,#fff1f2,#ffe4e6);
  border: 2px solid #fca5a5;
  border-radius: 22px; padding: 28px;
  margin-bottom: 20px;
}
.ood-icon   { font-size: 60px; line-height: 1; }
.ood-title  { font-size: 26px; font-weight: 900; color: #991b1b; margin-bottom: 6px; }
.ood-subtitle { font-size: 15px; color: #b91c1c; line-height: 1.55; }

.supported-box {
  background: #f0fdf4; border: 1px solid #86efac;
  border-radius: 18px; padding: 22px; margin-bottom: 18px;
}
.supported-box h3 { margin: 0 0 14px 0 !important; color: #14532d; font-size: 14px !important; font-weight: 800 !important; }
.plant-pills { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }
.plant-pill {
  background: white; border: 1px solid #86efac;
  color: #166534; padding: 8px 14px;
  border-radius: 999px; font-size: 14px; font-weight: 700;
}
.supported-note { margin: 0 !important; color: #166534; font-size: 13.5px; line-height: 1.65; }

.ood-reasons {
  background: #fff7ed; border: 1px solid #fdba74;
  border-radius: 18px; padding: 22px; margin-bottom: 18px;
}
.ood-reasons h3 { margin: 0 0 12px 0 !important; color: #9a3412; font-size: 14px !important; font-weight: 800 !important; }
.ood-reasons ul { margin: 0 0 12px 0 !important; padding-left: 20px; }
.ood-reasons li { color: #7c2d12; font-size: 13.5px; line-height: 1.65; margin-bottom: 8px; }
.ood-criteria-note { margin: 0 !important; font-size: 12.5px; color: #92400e; }

.ood-tips {
  background: #eff6ff; border: 1px solid #93c5fd;
  border-radius: 18px; padding: 22px; margin-bottom: 18px;
}
.ood-tips h3 { margin: 0 0 12px 0 !important; color: #1e3a8a; font-size: 14px !important; font-weight: 800 !important; }
.ood-tips ul { margin: 0 !important; padding-left: 20px; }
.ood-tips li { color: #1e40af; font-size: 13.5px; line-height: 1.65; margin-bottom: 8px; }

/* ── Normal result ── */
.result-card { animation: fadeUp 0.45s ease; }
@keyframes fadeUp { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }

.score-big {
  text-align: center; border: 1px solid;
  border-radius: 26px; padding: 26px 20px; margin-bottom: 16px;
}
.score-label { font-size:13px; text-transform:uppercase; letter-spacing:1.2px; font-weight:800; opacity:.85; }
.score-value { font-size:76px; font-weight:900; line-height:1; margin-top:8px; letter-spacing:-3px; }

.agreement-box { padding:18px 20px; border-radius:20px; margin-bottom:16px; border:1px solid; }
.agreement-box p { margin:6px 0 0 0 !important; color:#334155; line-height:1.55; font-size:14px; }
.agreement-score { font-size:17px; font-weight:900; }
.agreement-good    { background:#ecfdf5; border-color:#86efac; color:#166534; }
.agreement-warning { background:#fff7ed; border-color:#fdba74; color:#9a3412; }

.result-hero {
  display:flex; justify-content:space-between; align-items:center;
  gap:18px; padding:24px; border:1px solid; border-radius:22px; margin-bottom:18px;
}
.hero-left    { display:flex; align-items:center; gap:16px; }
.big-emoji    { font-size:58px; line-height:1; }
.overline     { color:#64748b; font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px; }
.result-hero h2 { margin:0 !important; color:#0f172a; font-size:26px !important; line-height:1.18 !important; font-weight:800 !important; }
.result-hero p  { margin:6px 0 0 0 !important; color:#475569; font-weight:600; }
.severity {
  display:inline-flex; align-items:center; gap:8px;
  padding:10px 14px; border:1px solid; border-radius:999px; white-space:nowrap;
}

.models-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:16px; }
.model-card  { border:1px solid #e2e8f0; background:white; border-radius:18px; padding:16px; }
.model-name  { color:#64748b; font-size:12px; text-transform:uppercase; letter-spacing:.9px; font-weight:800; margin-bottom:8px; }
.model-pred  { display:flex; gap:8px; align-items:center; color:#0f172a; font-weight:800; line-height:1.35; margin-bottom:10px; }
.mini-meta   { display:flex; align-items:center; gap:10px; }
.mini-meta span { border:1px solid; padding:6px 10px; border-radius:999px; font-size:12px; font-weight:800; }

.info-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:16px; }
.info-box  { border:1px solid #e2e8f0; border-radius:18px; padding:18px; background:#fff; }
.info-box h3 { margin:0 0 10px 0 !important; color:#065f46; font-size:14px !important; font-weight:800 !important; }
.info-box p  { margin:0 !important; color:#334155; font-size:14px; line-height:1.65; }

.disclaimer {
  margin-top:14px; padding:12px 14px;
  background:#f1f5f9; color:#475569;
  border-radius:14px; font-size:12.5px; line-height:1.6;
}

.footer-app {
  text-align:center; color:#64748b; font-size:13px;
  margin-top:24px; padding:20px;
}

@media (max-width:800px) {
  .main-header { padding:28px; }
  .main-header h1 { font-size:32px !important; }
  .score-value { font-size:58px; }
  .result-hero { flex-direction:column; align-items:flex-start; }
  .info-grid, .models-grid { grid-template-columns:1fr; }
  .ood-banner { flex-direction:column; }
}
"""


# ============================================================
# 9. INTERFACE GRADIO
# ============================================================
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="emerald",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    ),
    css=CSS,
    title="AgriScan AI — Diagnostic de feuilles",
) as demo:

    gr.HTML(f"""
    <div class="main-header">
      <div class="brand-pill">🌿 AgriScan AI · Accord 1/2 ou 2/2 · Rejet hors-base</div>
      <h1>Diagnostic intelligent<br>par double vérification</h1>
      <p>
        L'image est analysée par <b>EfficientNet-B0</b> et <b>ResNet-50</b>.
        Si la plante n'est pas reconnue (pas un poivron, pomme de terre ou tomate),
        un message de <b>rejet automatique</b> est affiché.
        Sinon, l'accord <b>2/2</b> ou <b>1/2</b> est indiqué sans pourcentage.
      </p>
      <div class="header-tags">
        <span class="header-tag">EfficientNet-B0</span>
        <span class="header-tag">ResNet-50</span>
        <span class="header-tag">🚫 Rejet hors-base</span>
        <span class="header-tag">Device : {DEVICE}</span>
      </div>
    </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, elem_classes="panel"):
            gr.HTML("<div class='section-title'>1. Image de la feuille</div>")
            img_input = gr.Image(
                type="pil",
                label="",
                show_label=False,
                height=390,
                sources=["upload", "webcam", "clipboard"],
            )
            analyze_btn = gr.Button(
                "🔍 Analyser avec les deux modèles",
                elem_classes="predict-btn"
            )
            clear_btn = gr.Button("Effacer", variant="secondary")
            gr.HTML("""
            <div class="upload-note">
              <b>⚠️ Plantes reconnues uniquement :</b><br>
              🫑 Poivron &nbsp;·&nbsp; 🥔 Pomme de terre &nbsp;·&nbsp; 🍅 Tomate<br><br>
              <b>Conseils pour une bonne image :</b><br>
              • Une seule feuille visible, nette et bien éclairée<br>
              • Éviter les ombres fortes<br>
              • Cadrer la feuille au centre
            </div>
            """)

        with gr.Column(scale=6, elem_classes="panel"):
            gr.HTML("<div class='section-title'>2. Résultat du diagnostic</div>")
            output_html = gr.HTML(value=EMPTY_HTML)

    gr.HTML("""
    <div class="footer-app">
      <b>AgriScan AI</b> · Classification de feuilles · PyTorch + Gradio ·
      EfficientNet-B0 + ResNet-50 · Rejet automatique hors-base
    </div>
    """)

    analyze_btn.click(fn=predict_leaf, inputs=img_input, outputs=output_html)
    img_input.change(fn=predict_leaf, inputs=img_input, outputs=output_html)
    clear_btn.click(
        fn=lambda: (None, EMPTY_HTML),
        inputs=None,
        outputs=[img_input, output_html]
    )


# ============================================================
# 10. LANCEMENT
# ============================================================
if __name__ == "__main__":
    demo.launch(share=True, debug=True)
