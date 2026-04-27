import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import warnings
warnings.filterwarnings('ignore')

columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

print("[*] Loading dataset...")
train = pd.read_csv("KDDTrain+.txt", names=columns)
test  = pd.read_csv("KDDTest+.txt",  names=columns)

train["label"] = train["label"].apply(lambda x: 0 if x == "normal" else 1)
test["label"]  = test["label"].apply(lambda x: 0 if x == "normal" else 1)

drop_cols = ["protocol_type","service","flag","label","difficulty"]
X_train = train.drop(columns=drop_cols).values
X_test  = test.drop(columns=drop_cols).values
y_test  = test["label"].values

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
X_normal = X_train[train["label"].values == 0]

print(f"[*] Training on {len(X_normal)} normal samples...")
dim = X_normal.shape[1]
inp = Input(shape=(dim,))
x   = Dense(32, activation='relu')(inp)
x   = Dense(16, activation='relu')(x)
x   = Dense(8,  activation='relu')(x)
x   = Dense(16, activation='relu')(x)
x   = Dense(32, activation='relu')(x)
out = Dense(dim, activation='sigmoid')(x)

autoencoder = Model(inp, out)
autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(X_normal, X_normal,
    epochs=30, batch_size=256, validation_split=0.1, verbose=1)

recon     = autoencoder.predict(X_normal)
mse_train = np.mean(np.power(X_normal - recon, 2), axis=1)
threshold = float(np.percentile(mse_train, 95))
print(f"[*] Threshold: {threshold:.6f}")

recon_test = autoencoder.predict(X_test)
mse_test   = np.mean(np.power(X_test - recon_test, 2), axis=1)
y_pred     = (mse_test > threshold).astype(int)

print(classification_report(y_test, y_pred, target_names=["Normal","Attack"]))
print(confusion_matrix(y_test, y_pred))

df_out = test.copy()
df_out["reconstruction_error"] = mse_test
df_out["predicted_anomaly"]    = y_pred
df_out.to_csv("anomaly_results.csv", index=False)

plt.figure(figsize=(10,4))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Training Loss'); plt.legend()
plt.savefig("training_loss.png")

plt.figure(figsize=(10,4))
plt.hist(mse_test[y_test==0], bins=100, alpha=0.6, label='Normal', color='blue')
plt.hist(mse_test[y_test==1], bins=100, alpha=0.6, label='Attack', color='red')
plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
plt.title('Reconstruction Error'); plt.legend()
plt.savefig("reconstruction_error.png")
print("[+] Done!")