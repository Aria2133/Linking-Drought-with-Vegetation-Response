# Google Earth Engine Scripts
   
   This folder contains my Google Earth Engine code for processing satellite data.
```
5. Scroll down
6. Commit message: `Create GEE scripts folder`
7. Click **"Commit new file"**

**Now you have a folder!** ✅

#### 2.3 Upload Your .js Files
1. Click on the **"gee_scripts"** folder you just created
2. Click **"Add file"** → **"Upload files"**
3. **Drag and drop** your .js files from your computer
   - Or click "choose your files" and select them
4. All your files will show up in the list
5. Scroll down
6. Commit message: `Add GEE scripts`
7. Click **"Commit changes"**

**DONE!** Your GEE code is now on GitHub! 🎉

---

## 📸 **VISUAL GUIDE**

### What You're Doing:
```
STEP 1: GET CODE FROM GEE
┌─────────────────────────┐
│  Google Earth Engine    │
│  (your scripts)         │
└───────────┬─────────────┘
            │ Copy & Paste
            ▼
┌─────────────────────────┐
│  Notepad                │
│  (save as .js files)    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Your Computer          │
│  (Desktop/Downloads)    │
└─────────────────────────┘

STEP 2: UPLOAD TO GITHUB
┌─────────────────────────┐
│  Your Computer          │
│  (.js files)            │
└───────────┬─────────────┘
            │ Upload
            ▼
┌─────────────────────────┐
│  GitHub Repository      │
│  (in gee_scripts folder)│
└─────────────────────────┘
```

---

## 💡 **EXAMPLE**

Let's say you have a GEE script called "MODIS Download"

**What you do:**

1. **In GEE Code Editor:**
   - Click on "MODIS Download"
   - Select all (Ctrl+A)
   - Copy (Ctrl+C)

2. **In Notepad:**
   - Paste (Ctrl+V)
   - File → Save As
   - Save as: `modis_download.js`
   - Save to Desktop

3. **In GitHub:**
   - Go to your repository
   - Create folder: `gee_scripts/`
   - Upload `modis_download.js`
   - Done!

---

## 🎯 **QUICK CHECKLIST**

Do this for each GEE script:

- [ ] Open script in GEE Code Editor
- [ ] Copy all code (Ctrl+A, Ctrl+C)
- [ ] Paste in Notepad
- [ ] Save as `.js` file
- [ ] Upload to GitHub in `gee_scripts/` folder

---

## ⚠️ **IMPORTANT TIPS**

1. **File names must end with `.js`**
   - ✅ Good: `modis_data.js`
   - ❌ Bad: `modis_data.txt`
   - ❌ Bad: `modis_data`

2. **When saving in Notepad:**
   - Change "Save as type" to **"All Files"**
   - Otherwise it adds `.txt` at the end

3. **Use simple names:**
   - ✅ Good: `download_modis.js`
   - ✅ Good: `calculate_ndvi.js`
   - ❌ Avoid spaces or weird characters

---

## 🚀 **HOW LONG DOES THIS TAKE?**

- **If you have 5 scripts:** About 15 minutes total
- **If you have 10 scripts:** About 25 minutes total

**It's quick!**

---

## ❓ **COMMON QUESTIONS**

**Q: Do I need to download any software?**
**A:** No! Just use Notepad (comes with Windows) and your web browser.

**Q: What if I make a mistake?**
**A:** You can delete files on GitHub and re-upload. No problem!

**Q: Do I need ALL my GEE scripts?**
**A:** No! Just upload your main/important ones. 3-5 scripts is enough.

**Q: What if my script is really long?**
**A:** That's fine! Just copy-paste the whole thing.

---

## ✅ **WHAT YOU'LL HAVE WHEN DONE**
```
Your GitHub Repository:
│
├── gee_scripts/                    ← NEW FOLDER
│   ├── README.md                   ← Explains what's in here
│   ├── download_modis.js           ← Your GEE script 1
│   ├── calculate_ndvi.js           ← Your GEE script 2
│   ├── cloud_mask.js               ← Your GEE script 3
│   └── export_data.js              ← Your GEE script 4
│
└── [All your other files]
