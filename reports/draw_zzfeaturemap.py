import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np

# 1. 수동으로 설계한 ZZ Feature Map 함수 정의
def manual_zz_feature_map(x, wires):
    """라이브러리 버전 이슈를 방지하기 위해 수동으로 설계한 ZZ Feature Map"""
    # Layer 1: Hadamard 중첩 상태 생성
    for i in range(len(wires)):
        qml.Hadamard(wires=wires[i])
    
    # Layer 2: Single-qubit rotation (Z-axis)
    for i in range(len(wires)):
        qml.RZ(2.0 * x[i], wires=wires[i])
    
    # Layer 3: Two-qubit interactions (Entanglement)
    for i in range(len(wires)):
        for j in range(i + 1, len(wires)):
            qml.CNOT(wires=[wires[i], wires[j]])
            qml.RZ(2.0 * (np.pi - x[i]) * (np.pi - x[j]), wires=wires[j])
            qml.CNOT(wires=[wires[i], wires[j]])

# 2. 양자 장치 및 QNode 설정
n_qubits = 4 
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def circuit(inputs):
    manual_zz_feature_map(inputs, range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 3. 회로 시각화
sample_inputs = [0.1, 0.2, 0.3, 0.4] # 시각화를 위한 샘플 데이터
fig, ax = qml.draw_mpl(circuit, decimals=2, style="black_white")(sample_inputs)

fig.suptitle("Manual Construction of ZZ Feature Map Circuit", fontsize=15)
plt.show()

# 논문용 고해상도 저장
fig.savefig("manual_zz_feature_map.png", dpi=300, bbox_inches='tight')