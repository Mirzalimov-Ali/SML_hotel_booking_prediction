Project: client hotelni bookingini cancel qiladimi yoki qilmaydimi
Nima uchun kerak: hotel clientni oldindan cancel qilishini aniqlash uchun
Yani model "bu client katta chance bilan cancel qiladi" deb aytishi mumkin

```mermaid
flowchart LR

Start((Start)) --> Split[Split Data]

Split --> Fill[Fill Missing Values]
Fill --> FE[Feature Engineering]
FE --> Encode[Encoding]
Encode --> Log[Log Transformation]
Log --> Scale[Scaling]

Scale --> Base[Base Model Training]

Base --> FS{Feature Selection}

FS --> F1[Filter<br/>Correlation / Chi2]
FS --> F2[Wrapper<br/>RFE]
FS --> F3[Embedded<br/>Lasso / Trees]
FS --> F4[Model Importance]

F1 --> SMOTE[SMOTE]
F2 --> SMOTE
F3 --> SMOTE
F4 --> SMOTE

SMOTE --> Tune{Hyperparameter Tuning}

Tune --> T1[Grid Search]
Tune --> T2[Random Search]
Tune --> T3[Bayesian Optimization]
Tune --> T4[Optuna]
Tune --> T5[Hyperband]

T1 --> Final[Final Training]
T2 --> Final
T3 --> Final
T4 --> Final
T5 --> Final

Final --> Infer[Inference]
Infer --> End((End))
```
