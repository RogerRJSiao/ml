"""
交通事故資料清洗規則
- 欄位定義 COLUMN_MIGRATION
- 類別欄位英文代碼對照 CATEGORY_MAPS 
- 經緯度範圍、地點解析 regex 等常數，供 cleaner.py 使用
"""

# ============================================================
# 欄位名稱：常數名稱 = COL_ + 輸出英文欄名大寫；不輸出的欄位以描述性英文命名
#          （COL_ 開頭為單一欄位，*_COLS / *_COLUMNS 為欄位清單）
# ============================================================
#--原始欄位
COL_YEAR = "發生年度"                                           # 1
COL_MONTH = "發生月份"                                          # 2
COL_INCIDENT_DATE = "發生日期"                                  # 3
COL_INCIDENT_TIME = "發生時間"                                  # 4
COL_SEVERITY = "事故類別名稱"                                   # 5
COL_POLICE = "處理單位名稱警局層"                               # 6
COL_ADDRESS = "發生地點"                                        # 7
COL_WEATHER = "天候名稱"                                        # 8
COL_LIGHT = "光線名稱"                                          # 9
COL_RD_TYPE = "道路類別-第1當事者-名稱"                         # 10
COL_SPEED_LIMIT = "速限-第1當事者"                              # 11
COL_RD_MAINTYPE = "道路型態大類別名稱"                          # 12
COL_RD_SUBTYPE = "道路型態子類別名稱"                           # 13
COL_RD_INCIDENT_MAIN = "事故位置大類別名稱"                     # 14
COL_RD_INCIDENT_SUB = "事故位置子類別名稱"                      # 15
COL_RD_SURFACE = "路面狀況-路面鋪裝名稱"                        # 16
COL_RD_SLIPPERY = "路面狀況-路面狀態名稱"                       # 17
COL_RD_DEFECT = "路面狀況-路面缺陷名稱"                         # 18
COL_RD_OBSTACLE = "道路障礙-障礙物名稱"                         # 19
COL_RD_VISION_QUALITY = "道路障礙-視距品質名稱"                 # 20
COL_RD_VISION = "道路障礙-視距名稱"                             # 21
COL_RD_SIGNALS = "號誌-號誌種類名稱"                            # 22
COL_RD_SIGNALS_DEFECT = "號誌-號誌動作名稱"                     # 23
COL_RD_CENTERLINE = "車道劃分設施-分向設施大類別名稱"           # 24
COL_RD_CENTERLINE_SUB = "車道劃分設施-分向設施子類別名稱"       # 25
COL_RD_FASTLANE_LINE = "車道劃分設施-分道設施-快車道或一般車道間名稱"  # 26
COL_RD_LANELINE = "車道劃分設施-分道設施-快慢車道間名稱"        # 27
COL_RD_SHOULDER = "車道劃分設施-分道設施-路面邊線名稱"          # 28
COL_ACC_TYPE = "事故類型及型態大類別名稱"                       # 29
COL_ACC_SUBTYPE = "事故類型及型態子類別名稱"                    # 30
COL_MAIN_SUSPICION_TYPE = "肇因研判大類別名稱-主要"             # 31
COL_MAIN_SUSPICION = "肇因研判子類別名稱-主要"                  # 32
COL_CASUALTY = "死亡受傷人數"                                   # 33
COL_PARTY_ORDER = "當事者順位"                                  # 34
COL_VECTOR = "當事者區分-類別-大類別名稱-車種"                  # 35
COL_VECTORTYPE = "當事者區分-類別-子類別名稱-車種"              # 36
COL_GENDER = "當事者屬-性-別名稱"                               # 37
COL_AGE = "當事者事故發生時年齡"                                # 38
COL_PROTECTION = "保護裝備名稱"                                 # 39
COL_PHONE_USE = "行動電話或電腦或其他相類功能裝置名稱"          # 40
COL_STATUS_MAJOR = "當事者行動狀態大類別名稱"                   # 41
COL_STATUS_MINOR = "當事者行動狀態子類別名稱"                   # 42
COL_DAMAGE_MAIN = "車輛撞擊部位大類別名稱-最初"                 # 43
COL_DAMAGE = "車輛撞擊部位子類別名稱-最初"                      # 44
COL_DAMAGE_OTHER_MAIN = "車輛撞擊部位大類別名稱-其他"           # 45
COL_DAMAGE_OTHER = "車輛撞擊部位子類別名稱-其他"                # 46
COL_SUSPICION_TYPE = "肇因研判大類別名稱-個別"                  # 47
COL_SUSPICION = "肇因研判子類別名稱-個別"                       # 48
COL_HIT_AND_RUN = "肇事逃逸類別名稱-是否肇逃"                   # 49
COL_LONGITUDE = "經度"                                          # 50
COL_LATITUDE = "緯度"                                           # 51

#--衍生欄位(原始資料不存在，清洗時產生)
COL_PERIOD = "時段"                        # ← 發生時間
COL_IS_NEAR_HIGHWAY = "高快速公路附近"     # ← 發生地點、處理單位名稱警局層
COL_CITY = "縣市"                          # ← 發生地點
COL_CITY_DISTRICT = "縣市鄉鎮市區"         # ← 發生地點
COL_DEAD_NUM = "死亡人數"                  # ← 死亡受傷人數
COL_INJURED_NUM = "受傷人數"               # ← 死亡受傷人數
COL_STATUS = "當事者行動狀態"              # ← 當事者行動狀態大類別名稱 + 子類別名稱

# ============================================================
# 欄位 migration：每個欄位的層級、處理方式、輸出英文欄名
# ============================================================
CASE, PARTY = "case", "party"   # 案件層級 / 當事者層級（依順位展開為 party1_、party2_）
KEEP = "keep"            # 保留，改為英文欄名輸出
DROP = "drop"            # 不使用，去除
SPLIT = "split"          # 拆成衍生欄位後移除
MERGE = "merge"          # 與其他欄位合併成衍生欄位後移除
INTERNAL = "internal"    # 清洗流程中使用，用完即移除
DERIVE = "derive"        # 衍生欄位

COLUMN_MIGRATION = [
    #--原始資料欄位              層級    處理       輸出英文欄名          #  序號：說明
    (COL_YEAR,                  CASE,  KEEP,     "year"),               #  1
    (COL_MONTH,                 CASE,  KEEP,     "month"),              #  2
    (COL_INCIDENT_DATE,         CASE,  KEEP,     "incident_date"),      #  3
    (COL_INCIDENT_TIME,         CASE,  KEEP,     "incident_time"),      #  4
    (COL_SEVERITY,              CASE,  KEEP,     "severity"),           #  5
    (COL_POLICE,                CASE,  KEEP,     "police"),             #  6
    (COL_ADDRESS,               CASE,  KEEP,     "address"),            #  7
    (COL_WEATHER,               CASE,  KEEP,     "weather"),            #  8
    (COL_LIGHT,                 CASE,  KEEP,     "light"),              #  9
    (COL_RD_TYPE,               CASE,  KEEP,     "rd_type"),            # 10
    (COL_SPEED_LIMIT,           CASE,  DROP,     None),                 # 11
    (COL_RD_MAINTYPE,           CASE,  DROP,     None),                 # 12：為道路型態子類別的上層分類
    (COL_RD_SUBTYPE,            CASE,  KEEP,     "rd_subtype"),         # 13
    (COL_RD_INCIDENT_MAIN,      CASE,  DROP,     None),                 # 14：為事故位置子類別的上層分類
    (COL_RD_INCIDENT_SUB,       CASE,  KEEP,     "rd_incident_sub"),    # 15
    (COL_RD_SURFACE,            CASE,  KEEP,     "rd_surface"),         # 16
    (COL_RD_SLIPPERY,           CASE,  KEEP,     "rd_slippery"),        # 17
    (COL_RD_DEFECT,             CASE,  KEEP,     "rd_defect"),          # 18
    (COL_RD_OBSTACLE,           CASE,  KEEP,     "rd_obstacle"),        # 19
    (COL_RD_VISION_QUALITY,     CASE,  DROP,     None),                 # 20：與視距名稱的「良好」完全重疊，只保留視距名稱
    (COL_RD_VISION,             CASE,  KEEP,     "rd_vision"),          # 21
    (COL_RD_SIGNALS,            CASE,  KEEP,     "rd_signals"),         # 22
    (COL_RD_SIGNALS_DEFECT,     CASE,  KEEP,     "rd_signals_defect"),  # 23
    (COL_RD_CENTERLINE,         CASE,  KEEP,     "rd_centerline"),      # 24
    (COL_RD_CENTERLINE_SUB,     CASE,  DROP,     None),                 # 25
    (COL_RD_FASTLANE_LINE,      CASE,  DROP,     None),                 # 26
    (COL_RD_LANELINE,           CASE,  KEEP,     "rd_laneline"),        # 27
    (COL_RD_SHOULDER,           CASE,  KEEP,     "rd_shoulder"),        # 28
    (COL_ACC_TYPE,              CASE,  KEEP,     "acc_type"),           # 29
    (COL_ACC_SUBTYPE,           CASE,  KEEP,     "acc_subtype"),        # 30
    (COL_MAIN_SUSPICION_TYPE,   CASE,  DROP,     None),                 # 31
    (COL_MAIN_SUSPICION,        CASE,  DROP,     None),                 # 32：與順位1個別肇因 100% 相同
    (COL_CASUALTY,              CASE,  SPLIT,    None),                 # 33：→ 死亡人數、受傷人數
    (COL_PARTY_ORDER,           PARTY, INTERNAL, None),                 # 34：依順位展開後移除
    (COL_VECTOR,                PARTY, KEEP,     "vector"),             # 35
    (COL_VECTORTYPE,            PARTY, KEEP,     "vectortype"),         # 36
    (COL_GENDER,                PARTY, INTERNAL, None),                 # 37：判定非人類當事者後移除
    (COL_AGE,                   PARTY, KEEP,     "age"),                # 38
    (COL_PROTECTION,            PARTY, KEEP,     "protection"),         # 39
    (COL_PHONE_USE,             PARTY, DROP,     None),                 # 40
    (COL_STATUS_MAJOR,          PARTY, MERGE,    None),                 # 41：→ 當事者行動狀態
    (COL_STATUS_MINOR,          PARTY, MERGE,    None),                 # 42：→ 當事者行動狀態
    (COL_DAMAGE_MAIN,           PARTY, DROP,     None),                 # 43
    (COL_DAMAGE,                PARTY, KEEP,     "damage"),             # 44
    (COL_DAMAGE_OTHER_MAIN,     PARTY, DROP,     None),                 # 45
    (COL_DAMAGE_OTHER,          PARTY, DROP,     None),                 # 46
    (COL_SUSPICION_TYPE,        PARTY, DROP,     None),                 # 47
    (COL_SUSPICION,             PARTY, KEEP,     "suspicion"),          # 48
    (COL_HIT_AND_RUN,           PARTY, KEEP,     "hit_and_run"),        # 49
    (COL_LONGITUDE,             CASE,  KEEP,     "longitude"),          # 50：原始檔排在當事者欄位後，但屬案件鍵值
    (COL_LATITUDE,              CASE,  KEEP,     "latitude"),           # 51：同上
    #--衍生欄位
    (COL_PERIOD,                CASE,  DERIVE,   "period"),
    (COL_IS_NEAR_HIGHWAY,       CASE,  DERIVE,   "is_near_highway"),
    (COL_CITY,                  CASE,  DERIVE,   "city"),
    (COL_CITY_DISTRICT,         CASE,  DERIVE,   "city_district"),
    (COL_DEAD_NUM,              CASE,  DERIVE,   "dead_num"),
    (COL_INJURED_NUM,           CASE,  DERIVE,   "injured_num"),
    (COL_STATUS,                PARTY, DERIVE,   "status"),
]

#--由 COLUMN_MIGRATION 推導，不另外維護
RAW_COLUMNS = [c for c, _, action, _ in COLUMN_MIGRATION if action != DERIVE]   # 原始檔應有的 51 欄
DROP_COLS = [c for c, _, action, _ in COLUMN_MIGRATION if action == DROP]
#--資料清洗產出用的英文欄名；當事者欄位輸出時以 party1_ / party2_ 為前綴
CASE_RENAME = {c: en for c, level, _, en in COLUMN_MIGRATION if level == CASE and en}
PARTY_RENAME = {c: en for c, level, _, en in COLUMN_MIGRATION if level == PARTY and en}

#--判斷是否為同一案件的規則：日期＋時間＋警局＋經緯度＋發生地點
CASE_KEY = [COL_INCIDENT_DATE, COL_INCIDENT_TIME, COL_POLICE, COL_LONGITUDE, COL_LATITUDE, COL_ADDRESS]

#--當事者欄位依順位填入前綴
PARTY_PREFIXES = ("順位1_", "順位2_")   # 當事者欄位依順位展開後的前綴

# ============================================================
# 資料型別
# ============================================================
#--數值欄位
NUMERIC_COLS = [COL_YEAR, COL_MONTH, COL_DEAD_NUM, COL_INJURED_NUM,
                COL_PARTY_ORDER, COL_AGE, COL_LONGITUDE, COL_LATITUDE]

# ============================================================
# 校正資料內容
# ============================================================
#--依「當事者屬-性-別名稱」判定，空白的當事者類別欄位（PARTY_CAT_COLS）補填標籤
NONHUMAN = ["無或物(動物、堆置物)"]      # 非人類當事者：年齡標 0，補填 NA_LABEL
NA_LABEL = "not_applicable"
UNSOLVED = ["肇事逃逸尚未查獲"]          # 肇逃未查獲：補填 UNSOLVED_LABEL
UNSOLVED_LABEL = "unsolved"
PARTY_CAT_COLS = [COL_VECTOR, COL_VECTORTYPE, COL_PROTECTION, COL_STATUS,
                  COL_DAMAGE, COL_SUSPICION, COL_HIT_AND_RUN]
#--錯字誤字
TYPO_FIX = {"未戴案全帽": "未戴安全帽"}

# ============================================================
# 類別欄位
# ============================================================
#--群組寫法(易於閱讀，但要補上helperz反轉翻讀)
SUSPICION_GROUPS = {  # party*_suspicion：肇因群組 → 原始值
    "yield": ["未依規定讓車", "左轉彎未依規定", "右轉彎未依規定", "迴轉未依規定",
              "起步未注意其他車(人)安全", "爭(搶)道行駛", "搶越行人穿越道"],
    "inattention": ["未注意車前狀態", "使用手持行動電話失控"],
    "distance": ["未保持行車安全距離", "未保持行車安全間隔"],
    "light": ["未依規定使用燈光", "夜間行駛無燈光設備"],
    "signal": ["違反號誌管制或指揮", "違反特定標誌(線)禁制", "搶(闖)越平交道"],
    "lane": ["變換車道或方向不當", "逆向行駛", "未靠右行駛", "違規超車", "蛇行、方向不定"],
    "speeding": ["超速失控", "未依規定減速"],
    "dui": ["酒醉(後)駕駛失控", "吸食違禁物後駕駛失控"],
    "fatigue": ["疲勞(患病)駕駛失控"],
    "pedestrian": ["未依規定行走行人穿越道、地下道、天橋而穿越道路",
                   "未依標誌、標線、號誌或手勢指揮穿越道路", "穿越道路未注意左右來車",
                   "橫越道路不慎", "在道路上嬉戲或奔走不定"],
    "passenger": ["頭手伸出車外而肇事", "上下車輛未注意安全", "未待車輛停妥而上下車", "乘坐不當而跌落"],
    "parking": ["開啟車門不當而肇事", "違規停車或暫停不當而肇事",
                "停車操作時，未注意其他車(人)安全", "倒車未依規定", "未待乘客安全上下開車",
                "暗處停車無燈光、標識", "拋錨未採安全措施"],
    "vehicle_defect": ["煞車失靈", "車輪脫落或輪胎爆裂", "車輛零件脫落", "方向操縱系統故障",
                       "燈光系統故障", "其他引起事故之故障"], 
    "vehicle_overload": ["裝載貨物不穩妥", "其他裝載不當肇事","裝載未盡安全措施", "裝卸貨物不當", "貨物超長、寬、高而肇事",
                       "載貨超重而失控", "超載人員而失控"],
    "environment": ["動物竄出", "路況危險無安全(警告)設施", "交通管制設施失靈或損毀",
                    "交通指揮不當", "其他交通管制不當", "在路上工作未設適當標識"],
    "unknown": ["不明原因肇事"],
    "no_fault": ["尚未發現肇事因素"],
    "others": ["其他引起事故之違規或不當行為", "其他引起事故之疏失或行為", "平交道看守疏失或未放柵欄"],
}
SUSPICION_MAP = {v: g for g, vals in SUSPICION_GROUPS.items() for v in vals}
SUSPICION_MAP[NA_LABEL] = NA_LABEL
SUSPICION_MAP[UNSOLVED_LABEL] = UNSOLVED_LABEL

#--直接寫法
CATEGORY_MAPS = {
    #--環境
    COL_WEATHER: {  # 8  weather
        "晴": "sunny", "陰": "cloudy", "雨": "rainy", "暴雨": "rainy",
        "強風": "windy", "風沙": "windy", "霧或煙": "foggy", "雪": "snowy"},
    COL_LIGHT: {  # 9  light
        "日間自然光線": "natural",
        "夜間(或隧道、地下道、涵洞)有照明": "artificial",
        "晨或暮光": "twilight",
        "夜間(或隧道、地下道、涵洞)無照明": "dark",
        "有照明未開啟或故障": "dark"},

    #--道路
    COL_RD_TYPE: {  # 10 rd_type
        "市區道路": "urban", "村里道路": "village", "省道": "provincial",
        "縣道": "county", "鄉道": "county", "國道": "freeway",
        "專用道路": "others", "其他": "others"},
    COL_RD_SUBTYPE: {  # 13 rd_subtype
        "四岔路": "4way", "三岔路": "3way", "多岔路": "multiway", "直路": "straight",
        "彎曲路及附近": "curve_or_slope", "坡路": "curve_or_slope",
        "橋樑": "structure", "高架道路": "structure", "地下道": "structure",
        "隧道": "structure", "涵洞": "structure",
        "有遮斷器": "railroad", "無遮斷器": "railroad",
        "巷弄": "others", "圓環": "others", "廣場": "others", "其他": "others"},
    COL_RD_INCIDENT_SUB: {  # 15 rd_incident_sub
        "交叉路口內": "within_intersection", "機車停等區": "within_intersection",
        "機車待轉區": "within_intersection",
        "交叉口附近": "near_intersection", "行人穿越道": "near_intersection",
        "人行道": "near_intersection", "穿越道附近": "near_intersection",
        "慢車道": "slow_lane", "機車優先道": "slow_lane", "機車專用道": "slow_lane",
        "快車道": "fast_lane", "一般車道(未劃分快慢車道)": "undivided_lane",
        "路肩、路緣": "shoulder", "公車專用道": "bus_lane",
        "其他": "others", "交通島(含槽化線)": "others", "迴轉道": "others",
        "直線匝道": "others", "環道匝道": "others", "減速車道": "others",
        "加速車道": "others", "收費站附近": "others"},

    #--路面狀況
    COL_RD_SURFACE: {  # 16 rd_surface
        "柏油": "asphalt", "水泥": "concrete",
        "其他鋪裝": "others", "無鋪裝": "others", "碎石": "others"},
    COL_RD_SLIPPERY: {  # 17 rd_slippery
        "乾燥": "dry", "濕潤": "moist", "油滑": "slippery", "泥濘": "slippery", "冰雪": "slippery"},
    COL_RD_DEFECT: {  # 18 rd_defect
        "無缺陷": "no_defects", "路面鬆軟": "loose", "突出(高低)不平": "uneven", "有坑洞": "pothole"},

    #--道路障礙
    COL_RD_OBSTACLE: {  # 19 rd_obstacle
        "無障礙物": "no_obstacles", "路上有停車": "cars", "道路工事(程)中": "construction",
        "其他障礙物": "others", "有堆積物": "others"},
    COL_RD_VISION: {  # 21 rd_vision
        "良好": "good", "其他": "others", "建築物": "others", "路上停放車輛": "cars",
        "彎道": "curve_or_slope", "坡道": "curve_or_slope", "樹木、農作物": "plants"},

    #--號誌
    COL_RD_SIGNALS: {  # 22 rd_signals
        "無號誌": "no_signals", "行車管制號誌": "signals",
        "行車管制號誌(附設行人專用號誌)": "signals_for_pedestrian", "閃光號誌": "flashing"},
    COL_RD_SIGNALS_DEFECT: {  # 23 rd_signals_defect
        "無號誌": "no_signals", "正常": "normal", "無動作": "malfunction", "不正常": "malfunction"},

    #--車道劃分設施
    COL_RD_CENTERLINE: {  # 24 rd_centerline
        "無": "no", "中央分向島": "island",
        "雙向禁止超車線": "line", "行車分向線": "line", "單向禁止超車線": "line"},
    COL_RD_LANELINE: {  # 27 rd_laneline
        "未繪設快慢車道分隔線": "no", "快慢車道分隔線": "line",
        "寛式快慢車道分隔島(50公分以上)": "island",
        "窄式快慢車道分隔島(無柵欄)": "island", "窄式快慢車道分隔島(附柵欄)": "island"},
    COL_RD_SHOULDER: {  # 28 rd_shoulder
        "有": "Y", "無": "N"},

    #--事故類型
    COL_ACC_TYPE: {  # 29 acc_type
        "車與車": "two_vehicles", "人與汽(機)車": "vehicle_person",
        "汽(機)車本身": "vehicle", "平交道事故": "railroad"},
    COL_ACC_SUBTYPE: {  # 30 acc_subtype
        "側撞": "side", "路口交岔撞": "angle", "追撞": "rear_end",
        "同向擦撞": "sideswipe_same", "對向擦撞": "sideswipe_opposite",
        "對撞": "head_on", "倒車撞": "backing",
        "穿越道路中": "pedestrian_crossing",
        "同向通行中": "pedestrian_same_direction", "對向通行中": "pedestrian_opposite_direction",
        "佇立路邊(外)": "pedestrian_other", "衝進路中": "pedestrian_other",
        "在路上作業中": "pedestrian_other", "從停車後(或中)穿出": "pedestrian_other",
        "在路上嬉戲": "pedestrian_other",
        "路上翻車、摔倒": "fall", "衝出路外": "run_off_road",
        "撞路樹、電桿": "fixed_object", "撞護欄(樁)": "fixed_object", "撞交通島": "fixed_object",
        "撞號誌、標誌桿": "fixed_object", "撞橋樑、建築物": "fixed_object", "撞工程施工": "fixed_object",
        "撞收費亭": "fixed_object",
        "暫停位置不當": "parking",
        "撞動物": "animal",
        "正越過平交道中": "railroad", "衝過(或撞壞)遮斷器": "railroad", "在平交道內無法行動": "railroad",
        "撞非固定設施": "others", "其他": "others"},

    #--當事者（順位1、2 共用）
    COL_VECTOR: {  # 35 party*_vector
        "機車": "motorcycle", "小客車": "car", "小貨車(含客、貨兩用)": "light_truck",
        "慢車": "slow_vehicle", "人": "person", "大客車": "bus",
        "大貨車": "heavy_vehicle", "曳引車": "heavy_vehicle",
        "半聯結車": "heavy_vehicle", "全聯結車": "heavy_vehicle",
        "特種車": "others", "軍車": "others", "其他車": "others",
        NA_LABEL: NA_LABEL, UNSOLVED_LABEL: UNSOLVED_LABEL},
    COL_VECTORTYPE: {  # 36 party*_vectortype：保留 vector 沒有的區分
        "大型重型1(550C.C.以上)": "heavy_motorcycle", "大型重型2(250-550C.C.)": "heavy_motorcycle",
        "普通重型": "motorcycle", "普通輕型": "motorcycle", "小型輕型": "motorcycle",
        "微型電動二輪車": "e_scooter", "腳踏自行車": "bicycle", "電動輔助自行車": "e_bike",
        "自用": "private", "營業用": "commercial",
        "計程車": "taxi", "租賃車": "rental",
        "大客車": "bus", "自用大客車": "bus", "民營公車": "bus", "公營公車": "bus", "民營客運": "bus", "公營客運": "bus", "遊覽車": "bus",
        "行人": "pedestrian", "乘客": "passenger", "火車": "train",
        "其他人": "others", "人力車": "others", "獸力車": "others", "其他慢車": "others",
        "警備車": "others", "救護車": "others", "消防車": "others", "工程車": "others",
        "其他特種車": "others", "動力機械": "others", "農耕用車(或機械)": "others",
        "拼裝車": "others", "拖車(架)": "others", "其他車": "others", "小型車": "others",
        "載重車": "others",
        NA_LABEL: NA_LABEL, UNSOLVED_LABEL: UNSOLVED_LABEL},
    COL_PROTECTION: {  # 39 party*_protection
        "戴安全帽或繫安全帶(使用幼童安全椅)": "Y",
        "未戴安全帽或未繫安全帶(未使用幼童安全椅)": "N",
        "不明": "Unknown",
        "其他(行人、慢車駕駛人)": "others",
        "其他(無需使用保護裝備之人)": "others",
        NA_LABEL: NA_LABEL, UNSOLVED_LABEL: UNSOLVED_LABEL},
    COL_STATUS: {  # 41+42 party*_status
        "車的狀態-向前直行中": "straight", "車的狀態-左轉彎": "left_turn", "車的狀態-右轉彎": "right_turn",
        "車的狀態-迴轉或橫越道路中": "u_turn_or_crossing",
        "車的狀態-向左變換車道": "lane_change", "車的狀態-向右變換車道": "lane_change",
        "車的狀態-插入行列": "lane_change", "車的狀態-超車(含超越)": "overtaking",
        "車的狀態-起步": "starting", "車的狀態-不明": "unknown", 
        "車的狀態-急減速或急停止": "stopping", "車的狀態-等待(引擎未熄火)": "stopping",
        "車的狀態-靜止(引擎熄火)": "stopping", "車的狀態-停車操作中": "stopping",
        "車的狀態-倒車": "reversing",
        "車的狀態-其他": "vehicle_other", "車的狀態": "vehicle_other",
        "人的狀態-步行": "pedestrian_moving", "人的狀態-奔跑": "pedestrian_moving",
        "人的狀態-靜立(止)": "pedestrian_static", "人的狀態-上、下車": "pedestrian_static",
        "人的狀態-其他": "pedestrian_other", "人的狀態-不明": "pedestrian_other",
        "人的狀態-向前直行中": "pedestrian_other", "人的狀態-右轉彎": "pedestrian_other",   # 疑似登錄錯誤
        NA_LABEL: NA_LABEL, UNSOLVED_LABEL: UNSOLVED_LABEL},
    COL_SUSPICION: SUSPICION_MAP,  # 48 party*_suspicion
    COL_HIT_AND_RUN: {  # 49 party*_hit_and_run
        "否": "N", "是": "Y", NA_LABEL: NA_LABEL, UNSOLVED_LABEL: UNSOLVED_LABEL},
}

# ============================================================
# 清洗規則
# ============================================================
#--定義時段
#--深夜、早晨、尖峰(上班)、白天、尖峰(下班)、晚上、深夜
PERIOD_BINS = [
    (0, 5, "late_night"), (5, 7, "early_morning"), (7, 9, "rush_hour"),
    (9, 17, "daytime"), (17, 19, "rush_hour"), (19, 22, "evening"), 
    (22, 24, "late_night"),
]

#--台澎金馬範圍
#--經度 min, 經度 max, 緯度 min, 緯度 max，落在任一範圍內才保留
GEO_BOXES = {
    "臺灣本島及附屬島嶼": (119.90, 122.10, 21.80, 25.40),
    "澎湖": (119.30, 119.80, 23.10, 23.85),
    "金門": (118.10, 118.55, 24.30, 24.60),
    "馬祖": (119.85, 120.55, 25.90, 26.40),
}

#--鄉鎮市區地名
#--第 4 字起，到第一個「鄉/鎮/市/區」為止
#--例外：地名第 2 字剛好是鄉鎮市區字者，需先列出，否則會被截斷（如「前鎮區」→「前鎮」）
TOWN_EXCEPTIONS = ["前鎮區", "平鎮區", "左鎮區", "新市區"]
TOWN_PATTERN = r"^(" + "|".join(TOWN_EXCEPTIONS) + r"|.{1,3}?[鄉鎮市區])"

#--高快速公路名稱
#--國道X號（含 國道一號、國道3甲、國3號、國一/國二橋下 等寫法）、台60～90線（含「臺」、省略「線」）、西濱快速道路
HIGHWAY_PATTERN = (
    r"國道\s*[0-9一二三四五六七八九十]+"              # 國道1號、國道一號、國道3甲、國道10號
    r"|國\d{1,2}號"                                    # 國3號
    r"|(?<![建復])國[一二三四五六七八九十](?![路街村巷])"  # 國二橋下、國三平面道路（排除 建國一路、復國二路）
    r"|[台臺](?:6\d|7\d|8\d|90)(?!\d)"                 # 台61線、臺64線、台66、台88快速公路
    r"|西濱快速"                                       # 西濱快速道路/公路（即台61線）
)
HIGHWAY_POLICE = "國道公路警察局"
