"""Name mapping between the dataset (eTryvoga), the map (geoBoundaries) and the live feed (ubilling)."""

# dataset name -> (geoBoundaries shapeName, ubilling Ukrainian name)
REGIONS = {
    "Cherkaska oblast": ("Cherkasy Oblast", "Черкаська область"),
    "Chernihivska oblast": ("Chernihiv Oblast", "Чернігівська область"),
    "Chernivetska oblast": ("Chernivtsi Oblast", "Чернівецька область"),
    "Dnipropetrovska oblast": ("Dnipropetrovsk Oblast", "Дніпропетровська область"),
    "Donetska oblast": ("Donetsk Oblast", "Донецька область"),
    "Ivano-Frankivska oblast": ("Ivano-Frankivsk Oblast", "Івано-Франківська область"),
    "Kharkivska oblast": ("Kharkiv Oblast", "Харківська область"),
    "Khersonska oblast": ("Kherson Oblast", "Херсонська область"),
    "Khmelnytska oblast": ("Khmelnytskyi Oblast", "Хмельницька область"),
    "Kirovohradska oblast": ("Kirovohrad Oblast", "Кіровоградська область"),
    "Kyiv City": ("Kyiv", "м. Київ"),
    "Kyivska oblast": ("Kyiv Oblast", "Київська область"),
    "Lvivska oblast": ("Lviv Oblast", "Львівська область"),
    "Mykolaivska oblast": ("Mykolaiv Oblast", "Миколаївська область"),
    "Odeska oblast": ("Odessa Oblast", "Одеська область"),
    "Poltavska oblast": ("Poltava Oblast", "Полтавська область"),
    "Rivnenska oblast": ("Rivne Oblast", "Рівненська область"),
    "Sumska oblast": ("Sumy Oblast", "Сумська область"),
    "Ternopilska oblast": ("Ternopil Oblast", "Тернопільська область"),
    "Vinnytska oblast": ("Vinnytsia Oblast", "Вінницька область"),
    "Volynska oblast": ("Volyn Oblast", "Волинська область"),
    "Zakarpatska oblast": ("Zakarpattia Oblast", "Закарпатська область"),
    "Zaporizka oblast": ("Zaporizhia Oblast", "Запорізька область"),
    "Zhytomyrska oblast": ("Zhytomyr Oblast", "Житомирська область"),
}
# Under a permanent alert since 2022 (Luhansk has only pre-April-2022 rows in the dataset): no model.
PERMANENT = {"Luhansk Oblast": "Luhanska oblast", "Autonomous Republic of Crimea": "Crimea", "Sevastopol": "Sevastopol"}

# Short display labels, e.g. "Cherkasy", "Kyiv obl."
LABELS = {r: shape.replace(" Oblast", "") for r, (shape, _) in REGIONS.items()}
LABELS.update({"Kyiv City": "Kyiv City", "Kyivska oblast": "Kyiv obl."})

UBILLING_TO_REGION = {uk: r for r, (_, uk) in REGIONS.items()}
