# 🧠 Detection of AI-Generated Arabic Text: A Data Mining Approach

مشروع يهدف إلى تطوير نموذج قادر على التمييز بين النصوص العربية المكتوبة بواسطة الإنسان وتلك المُولّدة آليًا باستخدام تقنيات الذكاء الاصطناعي، باستخدام تقنيات التنقيب في البيانات (Data Mining).

---

## 📊 Dataset

تم استخدام مجموعة البيانات من [KFUPM-JRCAI/arabic-generated-abstracts](https://huggingface.co/datasets/KFUPM-JRCAI/arabic-generated-abstracts)، والتي تحتوي على:

- **original_abstract**: ملخصات كتبها بشر.
- **{model}_generated_abstract**: ملخصات أنتجتها نماذج ذكاء اصطناعي (Allam، Jais، LLaMA، OpenAI).
- أنماط التوليد:
  - `by_polishing`
  - `from_title`
  - `from_title_and_content`

---

## 🧪 Project Phases

### ✅ Phase 1: Project Setup & Data Acquisition
- إنشاء هيكل مجلدات للمشروع.
- تحميل البيانات من Hugging Face.
- تحليل أولي للبيانات واستكشاف الفئات (AI vs Human).

### 🔧 Phase 2: Data Preprocessing
- تنظيف النصوص العربية: إزالة التشكيل، التكرار، الرموز، التنوين، وغيرها.
- التحقق من التوازن بين الفئات.

### 🔍 Phase 3: Feature Engineering
- تحويل النصوص إلى تمثيل رقمي باستخدام:
  - TF-IDF
  - Word Embeddings
  - أو BERT-based features (اختياري)

### 🧠 Phase 4: Modeling
- تصنيف ثنائي باستخدام نماذج:
  - Logistic Regression
  - Random Forest
  - SVM
  - (اختياري: Deep Learning)

### 📈 Phase 5: Evaluation & Interpretation
- تقييم النماذج باستخدام:
  - Accuracy, F1-score, Confusion Matrix
- تحليل الأخطاء.
- شرح الخصائص المؤثرة في التنبؤ.

---

## 📁 Folder Structure

```bash
arabic-text-detector/
│
├── data/
│   ├── raw/             # البيانات الأصلية
│   ├── processed/       # البيانات بعد المعالجة
│   └── external/        # أي بيانات خارجية
│
├── notebooks/           # Jupyter Notebooks للتجارب والاستكشاف
│
├── src/                 # سكربتات البايثون للمعالجة والنمذجة
│   ├── data_preparation.py
│   ├── modeling.py
│   ├── visualization.py
│   └── utils.py
│
├── models/              # النماذج المدربة (.pkl/.joblib)
├── reports/
│   ├── figures/         # الرسومات
│   └── presentations/   # العروض التقديمية
│
├── docs/                # الوثائق
├── README.md            # هذا الملف
├── requirements.txt     # مكتبات المشروع
└── .gitignore           # الملفات المستبعدة من Git
