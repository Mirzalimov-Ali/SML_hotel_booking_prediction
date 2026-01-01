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

FS --> F1[Filter<br/>Correlation]
FS --> F2[Wrapper<br/>RFE]
FS --> F3[Embedded<br/>Lasso / Trees]

F1 --> SMOTE[SMOTE]
F2 --> SMOTE
F3 --> SMOTE

SMOTE --> Tune{Hyperparameter Tuning}

Tune --> T1[Optuna]

T1 --> Final[Final Training]

Final --> Infer[Offline Testing]

Infer --> API[FastAPI Deployment]
API --> Client[Client / Hotel System]

Client --> End((End))
```