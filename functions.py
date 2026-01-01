import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import io
import zipfile
import unicodedata

### 자동화 관련 함수 선언
def _nfc(s: str) -> str:
    return unicodedata.normalize('NFC', s)

def _fix_zip_name(name: str) -> str:
    """
    zipfile이 cp437로 잘못 디코딩한 파일명을 복구 시도.
    1) cp437 bytes로 되돌린 뒤
    2) utf-8 / cp949 순으로 decode 시도
    """
    try:
        raw = name.encode("cp437")
    except Exception:
        return name

    for enc in ("utf-8", "cp949"):
        try:
            return raw.decode(enc)
        except Exception:
            pass

    # 최후: cp949로 깨지더라도 replace
    return raw.decode("cp949", errors="replace")


def _pick_excel_from_zip(z: zipfile.ZipFile, original_filename_no_ext: str):
    """ZIP 내부에서 원본 파일명 기반 매칭 -> 실패 시 첫 번째 엑셀 fallback"""

    infos = []
    for info in z.infolist():
        raw = info.filename
        fixed = _nfc(_fix_zip_name(raw)).replace("\\", "/")

        # __MACOSX 제거 + 엑셀만
        if fixed.startswith("__MACOSX") or "/__MACOSX/" in fixed:
            continue
        if not fixed.endswith((".xlsx", ".xls")):
            continue

        infos.append((info, fixed))

    # (표시용) clean name
    clean_names = []
    info_by_clean = {}
    for info, fixed in infos:
        base = fixed.split("/")[-1]
        clean_no_ext = base.rsplit(".", 1)[0]
        clean_names.append(clean_no_ext)
        info_by_clean[clean_no_ext] = info   # ✅ ZipInfo 저장

    # 1) 자동 매칭
    norm_orig = _nfc(original_filename_no_ext)
    for clean in clean_names:
        parts = [x for x in clean.split("_") if x]
        parts = [_nfc(x) for x in parts]
        if parts and all(part in norm_orig for part in parts):
            return clean, info_by_clean[clean], "matched"

    # 2) fallback: 첫 번째 엑셀
    if clean_names:
        clean = clean_names[0]
        return clean, info_by_clean[clean], "fallback_first"

    return None, None, "no_excel"



def prepare_batch_preview(alpha_file, original_filename_no_ext: str):
    """
    1) ZIP이면 매칭 후 batch_df 로드 / 엑셀이면 바로 로드
    2) 텍스트 미리보기 라인 생성
    return: (batch_df, meta, preview_lines, summary_lines)
    """
    meta = {
        "uploaded": alpha_file.name,
        "kind": "zip" if alpha_file.name.endswith(".zip") else "excel",
        "matched_file": None,
        "match_mode": None
    }

    # --- 1단계: 파일 확보 (업로드 즉시 실행) ---
    if alpha_file.name.endswith(".zip"):
        zip_bytes = io.BytesIO(alpha_file.getvalue())
        with zipfile.ZipFile(zip_bytes, 'r') as z:
            matched_clean, matched_info, mode = _pick_excel_from_zip(z, original_filename_no_ext)
            if mode == "no_excel":
                raise ValueError("ZIP 내부에 엑셀 파일이 없습니다.")

            meta["matched_file"] = matched_clean
            meta["match_mode"] = mode

            # ✅ 문자열 경로가 아니라 ZipInfo로 open
            with z.open(matched_info) as f:
                batch_df = pd.read_excel(
                f,
                dtype=str  # <─ 전체를 문자열로 받음 (숫자로 오인 금지)
            )

    else:
        meta["matched_file"] = alpha_file.name
        meta["match_mode"] = "no_match_needed"
        batch_df = pd.read_excel(
            alpha_file,
            dtype=str  # <─ 여기서도 동일
        )

    # --- 검증/정리 ---
    needed_cols = {"from", "to", "to_name", "alpha"}
    if not needed_cols.issubset(batch_df.columns):
        raise ValueError(f"엑셀 파일에 다음 컬럼이 포함되어야 합니다: {needed_cols}")


    df = batch_df.copy()
    df["from"] = df["from"].astype(str)
    df["to"] = df["to"].astype(str)
    df["to_name"] = df["to_name"].astype(str)
    df["to_name"] = df["to_name"].replace("nan", "").fillna("")
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")

    # alpha가 NaN인 행 제거
    df = df.dropna(subset=["alpha"])

    # --- 2단계: 텍스트 미리보기 생성 ---
    preview_lines = []
    for _, r in df.iterrows():
        nm = r["to_name"] if r["to_name"] else "-"
        preview_lines.append(f"{r['from']} -> {r['to']}({nm}) : {float(r['alpha'])*100:.4f}%")

    # from별 합/잔여
    summary_lines = []
    grouped = df.groupby("from")["alpha"].sum()
    for origin_code, total_alpha in grouped.items():
        remaining = 1.0 - float(total_alpha)
        summary_lines.append(
            f"[from={origin_code}] 이동합={float(total_alpha)*100:.4f}%, 잔여={remaining*100:.4f}%"
        )

    return df, meta, preview_lines, summary_lines

### 사용자 정의 함수 선언
def make_binary_matrix(matrix, threshold):
    # 임계값 이하의 원소들을 0으로 설정
    binary_matrix = matrix.apply(lambda x: np.where(x > threshold, 1, 0))
    return binary_matrix

def filter_matrix(matrix, threshold):
    # 임계값 이하의 원소들을 0으로 설정
    filtered_matrix = matrix.where(matrix > threshold, 0)
    return filtered_matrix

def calculate_network_centralities(G_bn, df_label, use_weight=False):
    weight_arg = 'weight' if use_weight else None

    # Degree
    in_degree_bn = dict(G_bn.in_degree(weight=weight_arg))
    out_degree_bn = dict(G_bn.out_degree(weight=weight_arg))

    df_degree = df_label.iloc[2:, :2].copy()
    df_degree["in_degree"] = pd.Series(in_degree_bn).sort_index().values.reshape(-1, 1)
    df_degree["out_degree"] = pd.Series(out_degree_bn).sort_index().values.reshape(-1, 1)

    gd_in_mean = df_degree["in_degree"].mean()
    gd_in_std = df_degree["in_degree"].std()
    gd_out_mean = df_degree["out_degree"].mean()
    gd_out_std = df_degree["out_degree"].std()

    # Betweenness
    bc_bn = nx.betweenness_centrality(G_bn, normalized=False, endpoints=False, weight=weight_arg)
    num_n = len(G_bn)
    bc_bn = {node: value / (num_n * (num_n - 1)) for node, value in bc_bn.items()}

    df_bc = df_label.iloc[2:, :2].copy()
    df_bc["Betweenness Centrality"] = pd.Series(bc_bn).sort_index().values.reshape(-1, 1)

    bc_mean = df_bc["Betweenness Centrality"].mean()
    bc_std = df_bc["Betweenness Centrality"].std()

    # Closeness
    cci_bn = nx.closeness_centrality(G_bn, distance=weight_arg)
    cco_bn = nx.closeness_centrality(G_bn.reverse(), distance=weight_arg)

    df_cc = df_label.iloc[2:, :2].copy()
    df_cc["Indegree_Closeness Centrality"] = pd.Series(cci_bn).sort_index().values.reshape(-1, 1)
    df_cc["Outdegree_Closeness Centrality"] = pd.Series(cco_bn).sort_index().values.reshape(-1, 1)

    cc_in_mean = df_cc["Indegree_Closeness Centrality"].mean()
    cc_in_std = df_cc["Indegree_Closeness Centrality"].std()
    cc_out_mean = df_cc["Outdegree_Closeness Centrality"].mean()
    cc_out_std = df_cc["Outdegree_Closeness Centrality"].std()

    # Eigenvector
    evi_bn = nx.eigenvector_centrality(G_bn, max_iter=500, tol=1e-06, weight=weight_arg)
    evo_bn = nx.eigenvector_centrality(G_bn.reverse(), max_iter=500, tol=1e-06, weight=weight_arg)

    df_ev = df_label.iloc[2:, :2].copy()
    df_ev["Indegree_Eigenvector Centrality"] = pd.Series(evi_bn).sort_index().values.reshape(-1, 1)
    df_ev["Outdegree_Eigenvector Centrality"] = pd.Series(evo_bn).sort_index().values.reshape(-1, 1)

    ev_in_mean = df_ev["Indegree_Eigenvector Centrality"].mean()
    ev_in_std = df_ev["Indegree_Eigenvector Centrality"].std()
    ev_out_mean = df_ev["Outdegree_Eigenvector Centrality"].mean()
    ev_out_std = df_ev["Outdegree_Eigenvector Centrality"].std()

    # HITS (가중치 지원 안 함 → 그대로 사용)
    hubs, authorities = nx.hits(G_bn, max_iter=1000, tol=1e-08, normalized=True)

    df_hi = df_label.iloc[2:, :2].copy()
    df_hi["HITS Hubs"] = pd.Series(hubs).sort_index().values.reshape(-1, 1)
    df_hi["HITS Authorities"] = pd.Series(authorities).sort_index().values.reshape(-1, 1)

    hi_hub_mean = df_hi["HITS Hubs"].mean()
    hi_hub_std = df_hi["HITS Hubs"].std()
    hi_ah_mean = df_hi["HITS Authorities"].mean()
    hi_ah_std = df_hi["HITS Authorities"].std()

    # Structural Hole Metrics (Constraint & Efficiency)
    constraints, efficiencies = calculate_kim_metrics(G_bn, weight=weight_arg)
    df_kim = df_label.iloc[2:, :2].copy()
    df_kim["Constraint"] = pd.Series(constraints).sort_index().values.reshape(-1, 1)
    df_kim["Efficiency"] = pd.Series(efficiencies).sort_index().values.reshape(-1, 1)

    # 평균(Mean) 및 표준편차(Std) 계산
    kim_const_mean = df_kim["Constraint"].mean()
    kim_const_std = df_kim["Constraint"].std()
    kim_eff_mean = df_kim["Efficiency"].mean()
    kim_eff_std = df_kim["Efficiency"].std()

    return (
        df_degree, df_bc, df_cc, df_ev, df_hi, df_kim,  # df_kim 추가
        gd_in_mean, gd_in_std, gd_out_mean, gd_out_std,
        bc_mean, bc_std,
        cc_in_mean, cc_in_std, cc_out_mean, cc_out_std,
        ev_in_mean, ev_in_std, ev_out_mean, ev_out_std,
        hi_hub_mean, hi_hub_std, hi_ah_mean, hi_ah_std,
        kim_const_mean, kim_const_std, kim_eff_mean, kim_eff_std  # 통계치 4개 추가
    )

@st.cache_data()
def get_submatrix_withlabel(df, start_row, start_col, end_row, end_col, first_index_of_df, numberoflabel = 2):
    row_indexs = list(range(first_index_of_df[0]-numberoflabel, first_index_of_df[0])) + list(range(start_row, end_row+1))
    col_indexs = list(range(first_index_of_df[1]-numberoflabel, first_index_of_df[1])) + list(range(start_col, end_col+1))
    # print(row_indexs)
    # print(col_indexs)

    submatrix_withlabel = df.iloc[row_indexs, col_indexs]
    return submatrix_withlabel

def reduce_negative_values(df, first_idx, mid_ID_idx):
    # 데이터프레임 복사
    df_editing = df.copy()

    # first_idx에서 mid_ID_idx까지의 범위 슬라이싱
    df_test = df_editing.iloc[first_idx[0]:mid_ID_idx[0], first_idx[1]:mid_ID_idx[1]].apply(pd.to_numeric, errors='coerce')

    # 음수 값이 있는 위치 추적 및 줄인 값 계산
    reduced_values_per_column = {}

    def reduce_and_track(value, col_index):
        if value < 0:
            # 줄일 값 저장 (음수 값의 절반)
            reduced_value = value / 2
            if col_index not in reduced_values_per_column:
                reduced_values_per_column[col_index] = 0
            reduced_values_per_column[col_index] += value - reduced_value  # 원래 값 - 절반으로 줄인 값
            return reduced_value
        return value

    # 음수인 값만 1/2로 줄이면서 추적
    for col_idx in range(df_test.shape[1]):
        df_test.iloc[:, col_idx] = df_test.iloc[:, col_idx].apply(lambda x: reduce_and_track(x, col_idx))

    # 수정된 값을 원본 데이터프레임에 다시 반영 (first_idx에서 mid_ID_idx까지의 부분)
    df_editing.iloc[first_idx[0]:mid_ID_idx[0], first_idx[1]:mid_ID_idx[1]] = df_test

    # 마지막 행에 줄인 값만큼 더하기
    last_row_index = df_editing.shape[0] - 1
    for col_idx, total_reduced in reduced_values_per_column.items():
        df_editing.iloc[last_row_index, first_idx[1] + col_idx] -= total_reduced

    msg = "음수 값들을 절반으로 줄이고, 줄인 값을 마지막 행에 더했습니다."

    # 중간 인덱스 값은 그대로 반환 (mid_ID_idx는 행과 열 인덱스이므로 이 경우 변경되지 않음)
    return df_editing, msg, mid_ID_idx




def get_mid_ID_idx(df, first_idx):
    matrix_X = df.iloc[first_idx[0]:, first_idx[1]:].astype(float)
    row_cnt, col_cnt, row_sum, col_sum = 0, 0, 0, 0
    for v in matrix_X.iloc[0]:
        if abs(row_sum - v) < 0.001:
            if v == 0:
                continue
            else: break
        row_cnt += 1
        row_sum += v
    for v in matrix_X.iloc[:, 0]:
        print(f'gap: {col_sum-v}, sum: {col_sum}, value: {v}')
        if abs(col_sum - v) < 0.001:
            if v == 0:
                continue
            else: break
        col_cnt += 1
        col_sum += v
    
    if row_cnt == col_cnt:
        size = row_cnt
    else:
        size = max(row_cnt, col_cnt)

    return (first_idx[0]+size, first_idx[1]+size)

def insert_row_and_col(df, first_idx, mid_ID_idx, code, name, num_of_label):
    df_editing = df.copy()
    df_editing.insert(loc=mid_ID_idx[1], column='a', value=np.nan, allow_duplicates=True)
    df_editing.iloc[first_idx[0]-num_of_label, mid_ID_idx[1]] = code
    df_editing.iloc[first_idx[0]-num_of_label+1, mid_ID_idx[1]] = name
    df_editing.iloc[first_idx[0]:, mid_ID_idx[1]] = 0
    df_editing.columns = range(df_editing.shape[1])
    df_editing = df_editing.T   
    df_editing.insert(loc=mid_ID_idx[0], column='a', value=np.nan, allow_duplicates=True)
    df_editing.iloc[first_idx[1]-num_of_label, mid_ID_idx[0]] = code
    df_editing.iloc[first_idx[1]-num_of_label+1, mid_ID_idx[0]] = name
    df_editing.iloc[first_idx[1]:, mid_ID_idx[0]] = 0
    df_editing.columns = range(df_editing.shape[1])
    df_editing = df_editing.T
    df_inserted = df_editing.copy()
    mid_ID_idx = (mid_ID_idx[0]+1, mid_ID_idx[1]+1)
    msg = f'A new row and column (Code: {code}, Name: {name}) have been inserted.'

    return df_inserted, mid_ID_idx, msg

def transfer_to_new_sector(df, first_idx, origin_code, target_code, ratio, code_label = 2):
    df_editing = df.copy()
    target_idx = df_editing.index[df_editing[first_idx[1]-code_label] == target_code].tolist()
    if len(target_idx) == 1:
        target_idx = target_idx[0]
    else:
        msg = 'ERROR: target code is not unique.'
        return df_editing, msg
    origin_idx = df_editing.index[df_editing[first_idx[1]-code_label] == origin_code].tolist()
    if len(origin_idx) == 1:
        origin_idx = origin_idx[0]
    else:
        msg = 'ERROR: origin code is not unique.'
        return df_editing, msg
    df_editing.iloc[first_idx[0]:, first_idx[1]:] = df_editing.iloc[first_idx[0]:, first_idx[1]:].apply(pd.to_numeric, errors='coerce')
    origin_idx = (origin_idx, origin_idx-first_idx[0]+first_idx[1])
    target_idx = (target_idx, target_idx-first_idx[0]+first_idx[1])
    df_editing.iloc[target_idx[0] ,first_idx[1]:] += df_editing.iloc[origin_idx[0] ,first_idx[1]:] * ratio
    df_editing.iloc[origin_idx[0] ,first_idx[1]:] = df_editing.iloc[origin_idx[0] ,first_idx[1]:] * (1-ratio)
    df_editing.iloc[first_idx[0]: ,target_idx[1]] += df_editing.iloc[first_idx[0]: ,origin_idx[1]] * ratio
    df_editing.iloc[first_idx[0]: ,origin_idx[1]] = df_editing.iloc[first_idx[0]: ,origin_idx[1]] * (1-ratio)

    msg = f'{ratio*100}% of {origin_code} has been moved to {target_code}.'
    return df_editing, msg

def remove_zero_series(df, first_idx, mid_ID_idx):
    df_editing = df.copy()
    df_test = df_editing.copy()
    df_test = df_editing.iloc[first_idx[0]:, first_idx[1]:].apply(pd.to_numeric, errors='coerce')
    zero_row_indices = df_test.index[(df_test == 0).all(axis=1)].tolist()
    zero_row_indices = [item for item in zero_row_indices if item>=first_idx[0] and item<=mid_ID_idx[0]]
    zero_col_indices = list(map(lambda x: x - first_idx[0] + first_idx[1], zero_row_indices))
    df_editing.drop(zero_row_indices, inplace=True)
    df_editing.drop(zero_col_indices, inplace=True, axis=1)
    df_editing.columns = range(df_editing.shape[1])
    df_editing.index = range(df_editing.shape[0])
    count = len(zero_col_indices)
    msg = f'{count}개의 행(열)이 삭제되었습니다.'
    mid_ID_idx = (mid_ID_idx[0] - count, mid_ID_idx[1] - count)
    return df_editing, msg, mid_ID_idx

def donwload_data(df, file_name):
    csv = convert_df(df)
    button = st.download_button(label=f"{file_name} 다운로드", data=csv, file_name=file_name+".csv", mime='text/csv')
    return button




@st.cache_data()
def load_data(file):
    st.session_state['df'] = pd.read_excel(file, header=None)
    return st.session_state['df']

@st.cache_data 
def convert_df(df):
    return df.to_csv(header=False, index=False).encode('utf-8-sig')


@st.cache_data
def make_zip_bytes(dfs: dict[str, pd.DataFrame]) -> bytes:
    """
    dfs: dict where keys are desired CSV filenames and values are DataFrames.
    반환값: ZIP 파일의 바이너리
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, df in dfs.items():
            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
            zf.writestr(f"{fname}.csv", csv_bytes)
    return buf.getvalue()

def download_multiple_csvs_as_zip(dfs: dict[str, pd.DataFrame], zip_name: str):
    zip_bytes = make_zip_bytes(dfs)
    return st.download_button(
        label=f"{zip_name} 다운로드",
        data=zip_bytes,
        file_name=f"{zip_name}.zip",
        mime="application/zip",
    )

def compute_leontief_inverse(A, epsilon=0.05, max_iter=100):
    """
    Leontief 역행렬을 무한급수(I + A + A^2 + ...)로 근사 계산하는 함수.
    수렴 조건: 누적합의 상대변화가 epsilon 이하가 될 때까지 반복.
    
    Parameters:
        A (ndarray): 투입계수행렬.
        epsilon (float): 수렴 판정 기준 (예: 0.05 = 5%).
        max_iter (int): 최대 반복 횟수 (무한급수의 수렴이 안 될 경우 대비).
    
    Returns:
        M (ndarray): I + A + A^2 + ... + A^r (r번째 항까지 계산한 근사 Leontief 역행렬).
    """
    n = A.shape[0]
    I = np.eye(n)           # n x n 항등행렬 생성
    M = I.copy()            # 초기 누적합: M(0) = I
    s_prev = np.sum(M)      # 초기 전체합 (s(0))
    k = 1                   # 거듭제곱 차수 초기화

    while k < max_iter:
        # A^k 계산 (행렬의 거듭제곱)
        A_power = np.linalg.matrix_power(A, k)
        
        # 누적합 업데이트: M(k) = M(k-1) + A^k
        M_new = M + A_power
        
        # 새로운 전체합 계산
        s_new = np.sum(M_new)
        
        # 상대 변화량 계산: (s(k) - s(k-1)) / s(k-1)
        ratio_change = (s_new - s_prev) / s_prev if s_prev != 0 else 0
        
        # 중간 결과 출력 (디버그용)
        print(f"Iteration {k}: ratio_change = {ratio_change:.4f}")
        
        # 수렴 판정: 상대 변화가 epsilon 이하이면 종료
        if ratio_change <= epsilon:
            M = M_new
            break
        
        # 업데이트 후 다음 반복 진행
        M = M_new
        s_prev = s_new
        k += 1
    
    return M

def separate_diagonals(N0):
    """
    입력 행렬 N0에서 대각원소와 비대각원소(네트워크 base)를 분리.
    
    Parameters:
        N0 (ndarray): Leontief 역행렬 근사 (I + A + A^2 + ...).
    
    Returns:
        Diagon (ndarray): N0에서 대각원소만 남기고 나머지를 0으로 만든 행렬.
        N (ndarray): N0에서 대각원소를 모두 0으로 만든 네트워크 행렬.
    """
    # np.diag: 대각 성분 추출, np.diagflat: 대각 행렬로 재구성
    Diagon = np.diag(np.diag(N0))
    N = N0 - Diagon
    return Diagon, N

def threshold_network(N, delta):
    """
    네트워크 행렬 N에서 임계치 delta보다 작은 값들을 0으로 대체.
    
    Parameters:
        N (ndarray): 원본 네트워크 행렬.
        delta (float): 임계치 값.
    
    Returns:
        N_thresholded (ndarray): thresholding 적용된 네트워크 행렬.
    """
    N_thresholded = N.copy()
    N_thresholded[N_thresholded < delta] = 0
    return N_thresholded

def create_binary_network(N):
    """
    가중치 네트워크 행렬 N를 이진(0-1) 네트워크로 변환 (양수면 1, 아니면 0).
    
    Parameters:
        N (ndarray): 가중치 네트워크 행렬.
    
    Returns:
        BN (ndarray): 이진화된 네트워크 (방향성 유지).
    """
    BN = (N > 0).astype(int)
    return BN

def create_undirected_network(BN):
    """
    방향성이 있는 이진 네트워크 BN를 무방향 네트워크로 변환.
    두 노드 간 어느 한쪽이라도 연결되어 있으면, 무방향 연결로 처리.
    
    Parameters:
        BN (ndarray): 이진화된 방향성 네트워크.
    
    Returns:
        UN (ndarray): 무방향(대칭) 이진 네트워크.
    """
    UN = ((BN + BN.T) > 0).astype(int)
    return UN

@st.cache_data()
def threshold_count(matrix):
    """
    [Integration Logic]
    1. Method 2 (Derivative): 변화율 안정화 지점 계산 (기존 유지)
    2. Method 2-1 (Distance): 원점 거리 최소화 지점 계산 (기존 유지 - 시작점 역할)
    3. Connectivity Check: Method 2-1 지점에서 고립 노드 발생 시, 사라질 때까지 Threshold 하향 조정 (신규 추가)
    """
    # -------------------------------------------------------------------------
    # 0. 데이터 준비
    # -------------------------------------------------------------------------
    if hasattr(matrix, 'to_numpy'):
        mat_data = matrix.to_numpy()
    else:
        mat_data = np.array(matrix)
        
    mat_data = mat_data.copy().astype(float)
    np.fill_diagonal(mat_data, 0) # 대각 성분 제외
    
    N = mat_data.shape[0]
    total_elements = N**2 - N
    
    # x축 설정
    delta = 0.01
    max_val = np.max(mat_data)
    x_values = np.arange(0, max_val + delta, delta)
    
    # -------------------------------------------------------------------------
    # 1. 지표 계산: y(생존율) & w(변화율)
    # -------------------------------------------------------------------------
    # y: Survival Ratio
    y_list = []
    for x in x_values:
        count = (mat_data >= x).sum()
        ratio = count / total_elements
        y_list.append(ratio)
    y = np.array(y_list)

    # w: Slope Change Rate (Method 2)
    if len(y) > 1:
        z = (y[1:] - y[:-1]) / delta
    else:
        z = np.zeros(len(y))

    w_list = []
    w_x_values = []
    for i in range(1, len(z)):
        val_w = abs(z[i] - z[i-1]) / delta 
        w_list.append(val_w)
        if i+1 < len(x_values):
            w_x_values.append(x_values[i+1])
    w = np.array(w_list)
    w_x_values = np.array(w_x_values)
    
    # Method 2: Stability Check (기존 로직 유지)
    epsilon = 0.01
    opt_idx_method2 = 0
    found_method2 = False
    
    for k in range(1, len(w)):
        if k > 3 and (w[k-1] - w[k]) <= epsilon:
            opt_idx_method2 = k + 2
            found_method2 = True
            break
    if not found_method2 and len(x_values) > 0:
        opt_idx_method2 = len(x_values) - 1
    
    threshold_method2 = x_values[opt_idx_method2] if len(x_values) > opt_idx_method2 else 0

    # -------------------------------------------------------------------------
    # 2. Method 2-1 (Distance Minimization) - [기준점]
    # -------------------------------------------------------------------------
    dist_sq = x_values**2 + y**2
    opt_idx_dist = np.argmin(dist_sq)
    
    threshold_dist = x_values[opt_idx_dist]
    min_y = y[opt_idx_dist] if len(y) > opt_idx_dist else 0

    # -------------------------------------------------------------------------
    # 3. [Logic Addition] Connectivity Backtracking
    # Method 2-1 지점(opt_idx_dist)에서 시작하여 0방향으로 스캔
    # -------------------------------------------------------------------------
    final_idx = opt_idx_dist
    adjusted = False
    
    # 현재 최적점(Distance Min)부터 0까지 역순 탐색
    for idx in range(opt_idx_dist, -1, -1):
        t = x_values[idx]
        
        # Binary Masking
        mask = (mat_data >= t) # 1 if connected, else 0
        
        # 고립 노드 체크 (Undirected 관점: In-degree + Out-degree == 0 이면 고립)
        # mask 행렬에서 행의 합(Out) + 열의 합(In) 계산
        degrees = mask.sum(axis=1) + mask.sum(axis=0)
        
        if np.any(degrees == 0):
            # 고립 노드가 존재함 -> Threshold가 너무 높음 -> 계속 낮춤(Loop Continue)
            continue
        else:
            # 고립 노드 없음 (All Connected) -> 멈춤
            final_idx = idx
            if idx < opt_idx_dist:
                adjusted = True
            break
    
    final_threshold = x_values[final_idx]
    final_y = y[final_idx] if len(y) > final_idx else 0

    # -------------------------------------------------------------------------
    # 4. 시각화 (모든 지표 포함)
    # -------------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 7))

    # [왼쪽 축] y(x) Curve
    color1 = 'tab:blue'
    ax1.set_xlabel('Threshold (x)')
    ax1.set_ylabel('Survival Ratio (y)', color=color1, fontweight='bold')
    ax1.plot(x_values, y, color=color1, label='y: Survival Ratio', linewidth=2, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # [오른쪽 축] w(t) Curve (기존 Method 2 시각화 유지)
    if len(w) > 0:
        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('Slope Change Rate (w)', color=color2, fontweight='bold')
        ax2.plot(w_x_values, w, color=color2, linestyle='--', alpha=0.5, label='w: Slope Stability')
        ax2.tick_params(axis='y', labelcolor=color2)

    # [Indicator 1] Method 2 (Stability) - 회색 수직선
    ax1.axvline(x=threshold_method2, color='gray', linestyle='-.', alpha=0.6,
                label=f'Method 2 (Stable): {threshold_method2:.4f}')

    # [Indicator 2] Method 2-1 (Distance Min) - 빨간 점 (원래의 수학적 최적점)
    ax1.plot(threshold_dist, min_y, 'ro', markersize=8, alpha=0.6,
             label=f'Method 2-1 (Dist Min): {threshold_dist:.4f}')

    # [Indicator 3] Final Decision (No Isolated) - 초록색 별/X (최종 결정)
    # 조정이 발생했다면 화살표와 함께 표시
    label_final = f'Final (No Isolated): {final_threshold:.4f}'
    
    if adjusted:
        # 조정된 경우: Method 2-1 -> Final 로 화살표 표시
        ax1.annotate('', xy=(final_threshold, final_y), xytext=(threshold_dist, min_y),
                     arrowprops=dict(arrowstyle="->", color='red', lw=2))
        ax1.plot(final_threshold, final_y, 'X', color='red', markersize=12, zorder=10, label=label_final)
    else:
        # 조정 안 된 경우: 빨간 점 위에 초록색 테두리 등을 씌워 강조
        ax1.plot(final_threshold, final_y, 'g*', markersize=14, zorder=10, label=label_final)

    # 범례 통합
    lines1, labels1 = ax1.get_legend_handles_labels()
    if len(w) > 0:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax1.legend(loc='upper right')

    plt.title('Threshold Optimization: Distance Min + Connectivity Check')
    fig.tight_layout()
    st.pyplot(fig)
    
    # -------------------------------------------------------------------------
    # 5. 결과 반환 및 설명
    # -------------------------------------------------------------------------
    msg_adjustment = ""
    if adjusted:
        msg_adjustment = f"⚠️ 수학적 최적점(`{threshold_dist:.4f}`)에서 고립 노드가 발견되어, `{final_threshold:.4f}`로 하향 조정했습니다."
    else:
        msg_adjustment = f"✅ 수학적 최적점(`{threshold_dist:.4f}`)이 고립 노드 없이 안정적입니다."

    st.markdown(f"""
    **최적 임계값 분석 결과**
    - **Method 2 (Stability)**: `{threshold_method2:.4f}`
    - **Method 2-1 (Dist Min)**: `{threshold_dist:.4f}` (Backtracking 시작점)
    - **Final Decision**: `{final_threshold:.4f}`
    
    {msg_adjustment}
    """)
    
    return final_threshold

def calculate_kim_metrics(G, weight='weight'):
    """
    Kim (2021) 방식의 Constraint와 Efficiency를 계산하여 딕셔너리로 반환
    Return: (constraints_dict, efficiencies_dict)
    """
    # 1. Constraint (Burt's constraint)
    # 가중치가 있으면 생산유발계수 등을 반영
    constraints = nx.constraint(G, weight=weight)
    
    # 2. Efficiency (Kim's redundancy-based)
    efficiencies = {}
    nodes = list(G.nodes())
    
    # 효율성 계산을 위한 사전 계산 (속도 최적화)
    # 양방향 거래량(volume) 계산 헬퍼
    def get_vol(u, v):
        if not G.has_edge(u, v): return 0.0
        return G[u][v].get(weight, 1.0) if weight else 1.0

    def get_bi_vol(u, v):
        return get_vol(u, v) + get_vol(v, u)

    node_total_volumes = {} # 분모: (In + Out sum)
    node_max_volumes = {}   # 분모: Max connection strength
    
    for n in nodes:
        # Total Volume (In + Out)
        vol_in = G.in_degree(n, weight=weight)
        vol_out = G.out_degree(n, weight=weight)
        node_total_volumes[n] = vol_in + vol_out
        
        # Max Volume with any partner
        partners = set(G.predecessors(n)) | set(G.successors(n))
        max_vol = 0.0
        for p in partners:
            vol = get_bi_vol(n, p)
            if vol > max_vol:
                max_vol = vol
        node_max_volumes[n] = max_vol

    # 개별 노드 효율성 계산
    for i in nodes:
        partners_i = list(set(G.predecessors(i)) | set(G.successors(i)))
        Ni = len(partners_i)
        
        if Ni == 0:
            efficiencies[i] = 0.0
            continue
            
        sum_Rij = 0.0
        for j in partners_i:
            # j와 i를 제외한 제3자(q) 탐색 (Redundancy check)
            potential_qs = [q for q in partners_i if q != j and q != i]
            
            R_ij = 0.0
            for q in potential_qs:
                # rho_iq: i의 전체 거래 중 q와의 비중
                vol_iq = get_bi_vol(i, q)
                denom_i = node_total_volumes.get(i, 0)
                rho_iq = vol_iq / denom_i if denom_i > 1e-9 else 0.0
                
                # tau_jq: j의 최대 거래 대비 q와의 강도
                vol_jq = get_bi_vol(j, q)
                max_vol_j = node_max_volumes.get(j, 0)
                tau_jq = vol_jq / max_vol_j if max_vol_j > 1e-9 else 0.0
                
                R_ij += (rho_iq * tau_jq)
            sum_Rij += R_ij
        
        # Kim's Efficiency Formula: epsilon = T_i / N_i where T_i = N_i - sum(R_ij)
        Ti = Ni - sum_Rij
        efficiencies[i] = Ti / Ni if Ni > 0 else 0.0
        
    return constraints, efficiencies

def calculate_standard_metrics(G_directed, weight='weight'):
    """Burt 표준 방식 (Efficiency = Effective Size / Out-Degree)"""
    std_constraints = nx.constraint(G_directed, weight=weight)
    effective_sizes = nx.effective_size(G_directed, weight=weight)
    
    std_efficiencies = {}
    for n, eff_size in effective_sizes.items():
        degree = G_directed.out_degree(n) # Standard Burt uses Out-degree for ego network size
        if degree > 0:
            std_efficiencies[n] = eff_size / degree
        else:
            std_efficiencies[n] = 0.0
            
    return std_constraints, std_efficiencies