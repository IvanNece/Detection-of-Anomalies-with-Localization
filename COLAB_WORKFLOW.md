# 🚀 Colab Workflow

## 📋 Prerequisiti
- Dataset su `My Drive/mvtecad/` (hazelnut, carpet, zipper)
- GitHub token: https://github.com/settings/tokens (scope: `repo`)

---

## 🔄 Workflow (5 Step)

### **1. Apri da VS Code**
- Click badge **"Open in Colab"** nel notebook

### **2. Esegui Notebook**
- `Runtime → Run all`
- Prima cella: setup (Drive mount, clone repo, dataset)
- Ultima cella: download ZIP risultati

### **3. Push da Colab**
```python
# Nella cella "Git Commit & Push"
GITHUB_TOKEN = "ghp_xxx"  # ← Il tuo token
GITHUB_EMAIL = "email@example.com"
GITHUB_NAME = "Ivan Nece"
```

### **4. Pull in Locale**
```powershell
git pull origin main
```
Scarica il notebook aggiornato da GitHub

### **5. Salva Output Localmente**
1. Estrai `notebook_XX_outputs.zip` (scaricato al step 2)
2. Copia file nella cartella progetto locale (es: `results/`, `outputs/`)
3. Commit e push:
```powershell
git add results/ outputs/
git commit -m "Add notebook XX outputs"
git push origin main
```

---

## ✅ Checklist Rapida
- [ ] Click badge Colab → Apri notebook
- [ ] Esegui tutte le celle
- [ ] Push notebook da Colab
- [ ] Pull in locale (`git pull`)
- [ ] Estrai ZIP scaricato
- [ ] Salva file in progetto locale
- [ ] Push file output (`git add/commit/push`)
