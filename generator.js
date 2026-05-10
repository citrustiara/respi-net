const {
    Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
    Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
    ShadingType, VerticalAlign, PageNumber, PageBreak
} = require('docx');
const fs = require('fs');

const PAGE_W = 11906;
const MARGIN = 1418;
const CONTENT_W = PAGE_W - 2 * MARGIN;  // 9070
const FONT = "Times New Roman";
const PRIO = "opcjonalny / niski / średni / wysoki / krytyczny";

const bThin = { style: BorderStyle.SINGLE, size: 2, color: "000000" };
const bHair = { style: BorderStyle.SINGLE, size: 1, color: "000000" };
const bNone = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const THIN = { top: bThin, bottom: bThin, left: bThin, right: bThin };
const HAIR = { top: bHair, bottom: bHair, left: bHair, right: bHair };
const NONE = { top: bNone, bottom: bNone, left: bNone, right: bNone };

// ── primitives ──────────────────────────────────────────────────────────────
const h1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const h3 = t => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const sp = (after = 160) => new Paragraph({ children: [new TextRun("")], spacing: { after } });
const br = () => new Paragraph({ children: [new PageBreak()] });

function body(text) {
    return new Paragraph({
        alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 120 },
        children: [new TextRun({ text, font: FONT, size: 22 })],
    });
}
function note(text) {
    return new Paragraph({
        alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 100 },
        indent: { left: 360 },
        children: [new TextRun({ text: `[${text}]`, font: FONT, size: 20, italics: true })],
    });
}

function tc(text, width, { bold = false, shade = "FFFFFF", sz = 20, border = HAIR, italic = false } = {}) {
    return new TableCell({
        borders: border,
        width: { size: width, type: WidthType.DXA },
        shading: { fill: shade, type: ShadingType.CLEAR },
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        verticalAlign: VerticalAlign.TOP,
        children: [new Paragraph({
            children: [new TextRun({ text, bold, italics: italic, font: FONT, size: sz })],
        })],
    });
}

// ── full-width spanning cell ─────────────────────────────────────────────────
function spanCell(text, shade = "C8C8C8", bold = true) {
    return new TableRow({
        children: [
            new TableCell({
                columnSpan: 2,
                borders: THIN,
                width: { size: CONTENT_W, type: WidthType.DXA },
                shading: { fill: shade, type: ShadingType.CLEAR },
                margins: { top: 60, bottom: 60, left: 100, right: 100 },
                children: [new Paragraph({
                    children: [new TextRun({ text, bold, font: FONT, size: 20 })],
                })],
            }),
        ]
    });
}

// ── record table: ID header + label|value rows ───────────────────────────────
function record(id, name, rows) {
    const C1 = 2600, C2 = CONTENT_W - C1;
    return new Table({
        width: { size: CONTENT_W, type: WidthType.DXA },
        columnWidths: [C1, C2],
        rows: [
            spanCell(`${id} ${name}`),
            ...rows.map((r, i) => new TableRow({
                children: [
                    tc(r.label, C1, { bold: true, shade: i % 2 === 0 ? "F0F0F0" : "FFFFFF" }),
                    tc(r.value, C2, { shade: i % 2 === 0 ? "F0F0F0" : "FFFFFF" }),
                ]
            })),
        ],
    });
}

// ── simple 2-col info table (cover, history) ─────────────────────────────────
function infoTable(rows) {
    const C1 = 3200, C2 = CONTENT_W - C1;
    return new Table({
        width: { size: CONTENT_W, type: WidthType.DXA },
        columnWidths: [C1, C2],
        rows: rows.map(r => new TableRow({
            children: [
                tc(r.label, C1, { bold: true, shade: "EFEFEF", border: THIN }),
                tc(r.value, C2, { border: THIN }),
            ]
        })),
    });
}

function historyTable() {
    const cols = [700, 2900, 1800, 2000, 1670];
    const hdrs = ["Wersja", "Opis modyfikacji", "Rozdział / strona", "Autor modyfikacji", "Data"];
    return new Table({
        width: { size: CONTENT_W, type: WidthType.DXA },
        columnWidths: cols,
        rows: [
            new TableRow({ tableHeader: true, children: hdrs.map((h, i) => tc(h, cols[i], { bold: true, shade: "C8C8C8", border: THIN })) }),
            new TableRow({
                children: [
                    tc("1.0", cols[0], { border: THIN }),
                    tc("Pierwsza wersja dokumentu – wypełnienie sekcji 1–5 (Zadanie 1 i 2)", cols[1], { border: THIN }),
                    tc("Wszystkie", cols[2], { border: THIN }),
                    tc("Maciej Łukasiewicz", cols[3], { border: THIN }),
                    tc("10.05.2026", cols[4], { border: THIN }),
                ]
            }),
        ],
    });
}

// ══════════════════════════════════════════════════════════════════════════════
//  DOCUMENT CONTENT
// ══════════════════════════════════════════════════════════════════════════════
const C = [];
const add = (...items) => items.forEach(x => C.push(x));

// ── Strona tytułowa ───────────────────────────────────────────────────────────
add(
    sp(1440),
    new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 480 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "000000", space: 6 } },
        children: [new TextRun({ text: "Specyfikacja wymagań systemowych", bold: true, font: FONT, size: 48 })],
    }),
    sp(480),
    infoTable([
        { label: "Nr zespołu:", value: "1" },
        { label: "Opiekun / Kierownik:", value: "dr hab. inż. Julian Szymański" },
        { label: "Nazwa projektu:", value: "Sieci neuronowe do analizy rytmu oddechowego" },
        { label: "Nazwa dokumentu:", value: "Specyfikacja wymagań systemowych" },
        { label: "Nr wersji:", value: "1.0" },
        { label: "Odpowiedzialny za dokument:", value: "Maciej Łukasiewicz" },
        { label: "Data pierwszego sporządzenia:", value: "10.05.2026" },
        { label: "Data ostatniej aktualizacji:", value: "10.05.2026" },
    ]),
    sp(480),
    body("Historia dokumentu"),
    sp(80),
    historyTable(),
    sp(480),
    body("Specyfikacja wymagań systemowych formalizuje wymagania klienta. Powinna być sformułowana w wyniku negocjacji zespołu projektowego z klientem."),
    br(),

    // ══════════════════════════════════════════════════════════════════════
    // 1. WPROWADZENIE
    // ══════════════════════════════════════════════════════════════════════
    h1("1. Wprowadzenie"),
    body("Niniejszy dokument stanowi specyfikację wymagań systemowych dla projektu inżynierskiego pt. \"Sieci neuronowe do analizy rytmu oddechowego\". Celem projektu jest opracowanie systemu pomiarowego umożliwiającego bezdotykowy pomiar parametrów witalnych człowieka – rytmu oddechowego oraz tętna – z wykorzystaniem czujników inercyjnych (IMU) i radaru mikrofalowego, a następnie zastosowanie sieci neuronowych do klasyfikacji i analizy zebranych sygnałów."),
    body("System składa się z układu sprzętowego opartego na mikrokontrolerze ESP32 (LilyGO T-Display) wyposażonym w moduł IMU LSM6DS3 (akcelerometr i żyroskop 6-osiowy) oraz radar doplerowski HB100 (10,525 GHz) z dedykowanym układem wzmacniacza i filtru aktywnego na bazie MCP6002. Dane z czujników przesyłane są przez USB/UART do komputera PC, gdzie podlegają przetwarzaniu sygnałów (filtracja Butterwortha, PCA, FFT/Welch) oraz analizie z użyciem modeli uczenia maszynowego."),
    body("Projekt jest realizowany jako praca dyplomowa inżynierska na Politechnice Gdańskiej. Autor: Maciej Łukasiewicz (197865). Promotor: dr hab. inż. Julian Szymański."),
    sp(),

    // ══════════════════════════════════════════════════════════════════════
    // 2. ŹRÓDŁA WYMAGAŃ
    // ══════════════════════════════════════════════════════════════════════
    h1("2. Źródła wymagań"),
    body("W tym rozdziale identyfikuje się źródła wymagań. Źródła wymagań mogą być osobowe (interesariusze) i nieosobowe (akty prawne, standardy, dokumentacja). Po zakończeniu specyfikowania wymagań trzeba sprawdzić, czy każde zidentyfikowane źródło podało przynajmniej jedno wymaganie."),
    sp(),

    h2("2.1. Interesariusze projektu"),
    note("Wymienić wszystkie osoby i instytucje zainteresowane projektem. Każdy interesariusz musi mieć priorytet. Osoby fizyczne podają kontakt. Instytucje podają reprezentanta i adres. Grupy podają sposób pozyskiwania wymagań."),
    sp(120),

    record("STKH_001", "Politechnika Gdańska – Wydział Elektroniki, Telekomunikacji i Informatyki", [
        { label: "Opis:", value: "Uczelnia będąca zleceniodawcą pracy dyplomowej inżynierskiej. Oczekuje wykonania systemu pomiarowego i analizy wyników jako wkładu w badania nad bezdotykowym monitoringiem witalnym." },
        { label: "Typ:", value: "osoba prawna" },
        { label: "Pełna nazwa:", value: "Politechnika Gdańska, Wydział Elektroniki, Telekomunikacji i Informatyki" },
        { label: "Adres:", value: "ul. Gabriela Narutowicza 11/12, 80-233 Gdańsk" },
        { label: "Reprezentant:", value: "dr hab. inż. Julian Szymański, julian.szymanski@pg.edu.pl" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("STKH_002", "Promotor pracy dyplomowej", [
        { label: "Opis:", value: "Opiekun naukowy projektu inżynierskiego. Określa wymagania merytoryczne, zatwierdza zakres pracy oraz ocenia wyniki badań i dokumentację." },
        { label: "Typ:", value: "osoba fizyczna" },
        { label: "Imię i nazwisko:", value: "dr hab. inż. Julian Szymański" },
        { label: "Kontakt:", value: "julian.szymanski@pg.edu.pl" },
        { label: "Reprezentuje:", value: "STKH_001 Politechnika Gdańska – WETI" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("STKH_003", "Badacz / Autor systemu", [
        { label: "Opis:", value: "Główny użytkownik systemu – student realizujący projekt. Przeprowadza eksperymenty, zbiera dane i trenuje modele sieci neuronowych." },
        { label: "Typ:", value: "osoba fizyczna" },
        { label: "Imię i nazwisko:", value: "Maciej Łukasiewicz (197865)" },
        { label: "Kontakt:", value: "maciej.lukasiewicz.student@pg.edu.pl" },
        { label: "Sposób pozyskania wymagań:", value: "Bezpośrednia analiza wymagań przez autora jako głównego dewelopera i użytkownika" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),
    sp(),

    record("STKH_004", "Badani ochotnicy (osoby mierzone)", [
        { label: "Opis:", value: "Osoby fizyczne poddające się pomiarom sygnałów witalnych w trakcie sesji rejestracji danych. Ich bezpieczeństwo i komfort są kluczowe; nie wchodzą w interakcję z oprogramowaniem bezpośrednio." },
        { label: "Typ:", value: "grupa osób" },
        { label: "Sposób pozyskania wymagań:", value: "Obserwacja, krótkie wywiady po sesjach pomiarowych" },
        { label: "Reprezentant:", value: "Maciej Łukasiewicz (koordynator sesji)" },
        { label: "Priorytet:", value: "wysoki" },
    ]),
    sp(),

    h2("2.2. Źródła nieosobowe"),
    note("Akty prawne, standardy, dokumentacja API itp. Podać dane bibliograficzne lub URL."),
    sp(120),

    record("RSRC_001", "RODO", [
        { label: "Opis:", value: "Rozporządzenie Parlamentu Europejskiego i Rady (UE) 2016/679 dotyczące ochrony danych osobowych. Dane zdrowotne rejestrowane od ochotników (sygnały IMU, radar) są danymi szczególnej kategorii i wymagają odpowiedniej ochrony." },
        { label: "Tytuł:", value: "Rozporządzenie (UE) 2016/679 (RODO / GDPR)" },
        { label: "Wydawca:", value: "Parlament Europejski i Rada UE" },
        { label: "Miejsce publikacji:", value: "UE, Bruksela" },
        { label: "Data publikacji:", value: "27.04.2016" },
        { label: "URL:", value: "https://eur-lex.europa.eu/legal-content/PL/TXT/?uri=CELEX:32016R0679" },
        { label: "Priorytet:", value: "wysoki" },
    ]),
    sp(),

    record("RSRC_002", "Dokumentacja Espressif ESP32 (ESP-IDF)", [
        { label: "Opis:", value: "Oficjalna dokumentacja techniczna mikrokontrolera ESP32 i frameworka ESP-IDF, stanowiąca podstawę implementacji firmware'u dla modułów IMU i ADC radaru." },
        { label: "Tytuł:", value: "ESP-IDF Programming Guide" },
        { label: "Wydawca:", value: "Espressif Systems" },
        { label: "Miejsce publikacji:", value: "online" },
        { label: "Data publikacji:", value: "2024 (aktualna wersja)" },
        { label: "URL:", value: "https://docs.espressif.com/projects/esp-idf/en/latest/esp32/" },
        { label: "Priorytet:", value: "wysoki" },
    ]),
    sp(),

    record("RSRC_003", "Karta katalogowa LSM6DS3 (STMicroelectronics)", [
        { label: "Opis:", value: "Dokumentacja techniczna modułu IMU LSM6DS3 (akcelerometr + żyroskop 6-osiowy). Definiuje zakresy pomiarowe, interfejs I2C, częstotliwości próbkowania i parametry elektryczne." },
        { label: "Tytuł:", value: "LSM6DS3 Datasheet" },
        { label: "Wydawca:", value: "STMicroelectronics" },
        { label: "Miejsce publikacji:", value: "online" },
        { label: "Data publikacji:", value: "2023" },
        { label: "URL:", value: "https://www.st.com/en/mems-and-sensors/lsm6ds3.html" },
        { label: "Priorytet:", value: "wysoki" },
    ]),
    sp(),

    record("RSRC_004", "Karta katalogowa HB100 (mikrofalowy moduł doplerowski)", [
        { label: "Opis:", value: "Dokumentacja techniczna radaru mikrofalowego HB100 (10,525 GHz). Określa parametry: czułość, zakres detekcji, poziom sygnału wyjściowego (IF ~5 mV), wymagania zasilania." },
        { label: "Tytuł:", value: "HB100 Microwave Motion Sensor Module Datasheet" },
        { label: "Wydawca:", value: "IC Station / producent modułu" },
        { label: "Miejsce publikacji:", value: "online" },
        { label: "Data publikacji:", value: "2023" },
        { label: "URL:", value: "https://www.icstation.com/hb100-doppler-microwave-module.html" },
        { label: "Priorytet:", value: "wysoki" },
    ]),
    sp(),
    br(),

    // ══════════════════════════════════════════════════════════════════════
    // 3. CELE SYSTEMU
    // ══════════════════════════════════════════════════════════════════════
    h1("3. Cele systemu"),
    body("Precyzuje się cele systemu z podziałem na biznesowe i funkcjonalne. Każdy cel biznesowy musi być wspierany przez przynajmniej jedno wymaganie. Celów biznesowych nie powinno być więcej niż 1–3; jeśli jest więcej niż jeden, ich priorytety muszą być różne."),
    sp(),

    h2("3.1. Cele biznesowe"),
    note("Korzyści związane z wdrożeniem systemu — materialne lub niematerialne. Realizacja pracy dyplomowej nie jest celem systemu."),
    sp(120),

    record("BSGL_001", "Nieinwazyjny monitoring parametrów witalnych", [
        { label: "Opis:", value: "Opracowanie i zademonstrowanie skutecznego, bezdotykowego systemu pomiaru rytmu oddechowego i tętna człowieka przy użyciu czujników IMU i radaru doplerowego. System ma umożliwić ciągły pomiar bez konieczności zakładania elektrod lub innych urządzeń dotykowych." },
        { label: "Źródło:", value: "STKH_002 Promotor pracy dyplomowej" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("BSGL_002", "Wkład naukowy w klasyfikację sygnałów biologicznych sieciami neuronowymi", [
        { label: "Opis:", value: "Opracowanie i ewaluacja modeli uczenia maszynowego (CNN/LSTM) zdolnych do klasyfikacji wzorców oddechowych i potencjalnej detekcji anomalii. Wyniki mają stanowić oryginalny wkład naukowy pracy dyplomowej." },
        { label: "Źródło:", value: "STKH_002 Promotor pracy dyplomowej" },
        { label: "Priorytet:", value: "wysoki" },
    ]),
    sp(),

    h2("3.2. Cele funkcjonalne"),
    note("Główne funkcjonalności systemu (maks. 10). Każdy cel funkcjonalny musi wspierać przynajmniej jeden cel biznesowy i być realizowany przez jedno lub wiele wymagań funkcjonalnych."),
    sp(120),

    record("FNGL_001", "Akwizycja danych z modułu IMU (LSM6DS3) przez ESP32", [
        { label: "Opis:", value: "System musi umożliwiać odczyt danych z akcelerometru i żyroskopu LSM6DS3 przez interfejs I2C, strumieniowanie ich do PC przez USB/UART i zapis do plików CSV." },
        { label: "Cel biznesowy:", value: "BSGL_001 Nieinwazyjny monitoring parametrów witalnych" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("FNGL_002", "Akwizycja danych z radaru doplerowego HB100 przez ESP32 ADC", [
        { label: "Opis:", value: "System musi umożliwiać wysoko-prędkościowy odczyt sygnału IF z radaru HB100 przez 12-bitowy ADC ESP32 (baud 921 600), zapis do pliku CSV i wizualizację na żywo." },
        { label: "Cel biznesowy:", value: "BSGL_001 Nieinwazyjny monitoring parametrów witalnych" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("FNGL_003", "Przetwarzanie sygnałów i ekstrakcja cech", [
        { label: "Opis:", value: "Oprogramowanie PC musi implementować potok DSP: PCA dla niezależności orientacji IMU, filtrację Butterwortha (pasma: oddech 0,1–0,5 Hz, tętno 0,65–4,0 Hz), transformatę FFT/Welch, detekcję pików." },
        { label: "Cel biznesowy:", value: "BSGL_001 Nieinwazyjny monitoring parametrów witalnych" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("FNGL_004", "Budowa i trening modeli sieci neuronowych", [
        { label: "Opis:", value: "System musi zapewnić potok przygotowania zbioru danych (labeling, normalizacja) i trenowanie modeli CNN/LSTM do klasyfikacji wzorców oddechowych." },
        { label: "Cel biznesowy:", value: "BSGL_002 Wkład naukowy w klasyfikację sygnałów" },
        { label: "Źródło:", value: "STKH_002 Promotor pracy dyplomowej" },
        { label: "Priorytet:", value: "wysoki" },
    ]),
    sp(),

    record("FNGL_005", "Wizualizacja i raportowanie wyników", [
        { label: "Opis:", value: "System musi generować wykresy dziedziny czasu i częstotliwości (PSD Welcha, FFT), metryki ewaluacji modeli (accuracy, F1) oraz pozwalać na eksport danych i wykresów." },
        { label: "Cel biznesowy:", value: "BSGL_002 Wkład naukowy w klasyfikację sygnałów" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "wysoki" },
    ]),
    sp(),
    br(),

    // ══════════════════════════════════════════════════════════════════════
    // 4. OTOCZENIE SYSTEMU
    // ══════════════════════════════════════════════════════════════════════
    h1("4. Otoczenie systemu"),
    body("Opisuje się kontekst, w jakim ma pracować system: użytkowników oraz systemy zewnętrzne, z którymi system będzie współdziałał bezpośrednio."),
    sp(),

    h2("4.1. Użytkownicy"),
    note("Każdy użytkownik reprezentuje rolę. Należy podać potrzeby i zadania użytkownika. Potrzeby to funkcjonalności, które system musi dostarczyć; zadania to działania, które użytkownik musi wykonać. Każdy użytkownik musi mieć przydzielone wymagania funkcjonalne."),
    sp(120),

    record("USER_001", "Badacz / Operator systemu pomiarowego", [
        { label: "Opis:", value: "Osoba (autor projektu) konfigurująca i obsługująca stanowisko pomiarowe: podłącza ESP32, uruchamia sesje akwizycji, monitoruje jakość sygnału na żywo i zapisuje dane." },
        { label: "Potrzeby:", value: "1. Uruchomienie akwizycji IMU i radaru w jednym poleceniu.\n2. Podgląd sygnału w czasie rzeczywistym (wykres czasu i FFT).\n3. Zapis danych do pliku CSV z automatycznym stemplem czasowym.\n4. Informacja o aktualnej częstotliwości próbkowania i wykrytym ruchu." },
        { label: "Zadania:", value: "1. Połączenie ESP32 z PC przez USB (COM port).\n2. Uruchomienie skryptu (radar_viewer.py lub script.py).\n3. Przeprowadzenie sesji pomiarowej (statyczna, dynamiczna).\n4. Zakończenie sesji i weryfikacja pliku CSV." },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("USER_002", "Analityk danych / Inżynier ML", [
        { label: "Opis:", value: "Osoba (autor projektu w roli analitycznej) przetwarzająca zebrane pliki CSV: stosująca DSP, budująca zbiory danych, trenująca i ewaluująca modele sieci neuronowych." },
        { label: "Potrzeby:", value: "1. Wczytanie i walidacja pliku CSV.\n2. Uruchomienie potoku DSP (PCA, filtracja, FFT).\n3. Eksport cech do formatu gotowego do trenowania.\n4. Wizualizacja i eksport wykresów." },
        { label: "Zadania:", value: "1. Wczytanie danych (pandas DataFrame).\n2. Przetwarzanie sygnałów (scipy, numpy).\n3. Trening modeli (PyTorch / TensorFlow).\n4. Generowanie raportów z wynikami." },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "wysoki" },
    ]),
    sp(),

    h2("4.2. Systemy zewnętrzne"),
    note("Systemy, z którymi projektowany system współdziała bezpośrednio. Podać interfejs (istniejący lub do zaprojektowania)."),
    sp(120),

    record("XSYS_001", "ESP32 (LilyGO T-Display) – firmware IMU", [
        { label: "Opis:", value: "Mikrokontroler realizujący akwizycję danych z LSM6DS3 przez I2C i transmisję ramek CSV przez UART/USB. Działa jako urządzenie peryferyjne w stosunku do oprogramowania PC." },
        { label: "Potrzeby:", value: "1. Odbiór ramek CSV w formacie: timestamp_ms, ax, ay, az, gx, gy, gz." },
        { label: "Zadania:", value: "1. Przesyłanie danych w czasie rzeczywistym z częstotliwością ≥200 Hz." },
        { label: "Interfejs:", value: "UART (USB CDC), baud 115 200, format CSV, newline-delimited" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("XSYS_002", "ESP32 – firmware ADC radaru (HB100)", [
        { label: "Opis:", value: "Drugi tryb pracy ESP32 (esp32_radar_adc): pobieranie próbek z 12-bitowego ADC GPIO34, budowanie ramek z timestampem i napięciem mV, transmisja z bardzo dużą prędkością." },
        { label: "Potrzeby:", value: "1. Odbiór ramek CSV: timestamp_ms, raw_adc, voltage_mV." },
        { label: "Zadania:", value: "1. Przesyłanie danych z próbkowaniem ≥500 Hz (baud 921 600)." },
        { label: "Interfejs:", value: "UART (USB CDC), baud 921 600, format CSV, newline-delimited" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("XSYS_003", "Radar HB100 z układem wzmacniacza MCP6002", [
        { label: "Opis:", value: "Moduł mikrofalowy 10,525 GHz z wyjściem IF ~5 mV. Dedykowany układ filtru aktywnego i wzmacniacza (MCP6002) kondycjonuje sygnał do zakresu ADC ESP32. System jest zależny od parametrów elektrycznych tego układu." },
        { label: "Potrzeby:", value: "1. Dostosowanie amplitudy sygnału IF do zakresu 0–3,3 V ADC." },
        { label: "Zadania:", value: "1. Wzmocnienie i filtracja pasmowa sygnału doplerowego." },
        { label: "Interfejs:", value: "Analogowy (napięciowy) – sygnał do GPIO34 ESP32" },
        { label: "Źródło:", value: "RSRC_004 Karta katalogowa HB100" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),
    br(),

    // ══════════════════════════════════════════════════════════════════════
    // 5. PRZEWIDYWANE KOMPONENTY SYSTEMU
    // ══════════════════════════════════════════════════════════════════════
    h1("5. Przewidywane komponenty systemu"),
    body("Wyszczególnienie komponentów systemu pomaga w uzyskaniu kompletności wymagań. Każdy komponent powinien mieć przypisane przynajmniej jedno wymaganie funkcjonalne."),
    sp(),

    h2("5.1. Podsystemy"),
    note("Komponenty złożone. Podać lokalizację (komponent sprzętowy) i listę składowych."),
    sp(120),

    record("SSYS_001", "Podsystem akwizycji danych IMU", [
        { label: "Opis:", value: "Realizuje odczyt 6-osiowych danych z LSM6DS3 przez I2C na ESP32 i transmisję ramek CSV przez USB/UART do PC." },
        { label: "Lokalizacja:", value: "HCMP_001 ESP32 LilyGO T-Display" },
        { label: "Komponenty:", value: "SCMP_001 Firmware ESP-IDF IMU, HCMP_002 Moduł LSM6DS3" },
        { label: "Powiązania:", value: "SSYS_002 Podsystem akwizycji radaru" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("SSYS_002", "Podsystem akwizycji danych radaru", [
        { label: "Opis:", value: "Realizuje wysoko-prędkościowy odczyt sygnału doplerowego z HB100 przez 12-bit ADC ESP32 (baud 921 600) i transmisję ramek CSV do PC." },
        { label: "Lokalizacja:", value: "HCMP_001 ESP32 LilyGO T-Display" },
        { label: "Komponenty:", value: "SCMP_002 Firmware ESP-IDF ADC radaru, HCMP_003 Moduł HB100 z wzmacniaczem" },
        { label: "Powiązania:", value: "SSYS_001 Podsystem akwizycji IMU" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("SSYS_003", "Podsystem przetwarzania sygnałów (PC)", [
        { label: "Opis:", value: "Oprogramowanie Python na PC: wczytuje CSV, stosuje PCA, filtry Butterwortha, FFT/Welch, detekcję pików i generuje wykresy." },
        { label: "Lokalizacja:", value: "HCMP_004 Komputer PC badacza" },
        { label: "Komponenty:", value: "SCMP_003 script.py (IMU DSP), SCMP_004 radar_viewer.py (radar DSP)" },
        { label: "Powiązania:", value: "SSYS_004 Podsystem ML, SSYS_001 Podsystem IMU, SSYS_002 Podsystem radaru" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("SSYS_004", "Podsystem uczenia maszynowego", [
        { label: "Opis:", value: "Potok ML: przygotowanie datasetu (labeling, normalizacja), trening modeli CNN/LSTM, ewaluacja (accuracy, F1), eksport modelu." },
        { label: "Lokalizacja:", value: "HCMP_004 Komputer PC badacza" },
        { label: "Komponenty:", value: "SCMP_005 Skrypt trenowania ML" },
        { label: "Powiązania:", value: "SSYS_003 Podsystem DSP" },
        { label: "Źródło:", value: "STKH_002 Promotor pracy dyplomowej" },
        { label: "Priorytet:", value: "wysoki" },
    ]),
    sp(),

    h2("5.2. Komponenty sprzętowe"),
    note("Urządzenia pełniące aktywną rolę w systemie (serwery, terminale itp.)."),
    sp(120),

    record("HCMP_001", "ESP32 LilyGO T-Display (mikrokontroler)", [
        { label: "Opis:", value: "Mikrokontroler Xtensa LX6 240 MHz, 4 MB Flash, Wi-Fi/BT, GPIO, ADC 12-bit, USB CDC. Uruchamia firmware IMU lub firmware ADC radaru." },
        { label: "Powiązania:", value: "HCMP_002 LSM6DS3, HCMP_003 HB100, HCMP_004 PC badacza" },
        { label: "Źródło:", value: "RSRC_002 Dokumentacja ESP-IDF" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("HCMP_002", "Moduł IMU LSM6DS3", [
        { label: "Opis:", value: "Akcelerometr + żyroskop 6-osiowy (STMicroelectronics). Interfejs I2C (SDA GPIO21, SCL GPIO22), zasilanie 3,3 V, częstotliwość próbkowania ≥208 Hz." },
        { label: "Powiązania:", value: "HCMP_001 ESP32" },
        { label: "Źródło:", value: "RSRC_003 Karta katalogowa LSM6DS3" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("HCMP_003", "Radar HB100 z układem wzmacniacza MCP6002", [
        { label: "Opis:", value: "Moduł mikrofalowy 10,525 GHz (Doppler). Sygnał IF ~5 mV wzmacniany dedykowanym aktywnym filtrem/wzmacniaczem na MCP6002 do poziomu 0–3,3 V. Wyjście podłączone do GPIO34 ESP32." },
        { label: "Powiązania:", value: "HCMP_001 ESP32" },
        { label: "Źródło:", value: "RSRC_004 Karta katalogowa HB100" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("HCMP_004", "Komputer PC badacza", [
        { label: "Opis:", value: "Komputer z systemem Windows/Linux, Python ≥3.10, biblioteki: pyserial, numpy, scipy, pandas, matplotlib, PyTorch/TensorFlow. Odbiera dane z ESP32 przez USB i przetwarza je." },
        { label: "Powiązania:", value: "HCMP_001 ESP32" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    h2("5.3. Komponenty programowe"),
    note("Programy, aplikacje, biblioteki stanowiące odrębną całość."),
    sp(120),

    record("SCMP_001", "Firmware ESP-IDF – moduł IMU (esp32_imu_stream)", [
        { label: "Opis:", value: "Firmware C/C++ (ESP-IDF) obsługujący LSM6DS3 przez I2C, pakujący dane 6-osiowe z timestampem i emitujący ramki CSV przez UART 115 200 baud." },
        { label: "Lokalizacja:", value: "HCMP_001 ESP32 LilyGO T-Display" },
        { label: "Powiązania:", value: "SCMP_003 script.py" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("SCMP_002", "Firmware ESP-IDF – moduł ADC radaru (esp32_radar_adc)", [
        { label: "Opis:", value: "Firmware C/C++ (ESP-IDF) próbkujący ADC GPIO34 z maksymalną prędkością, pakujący ramki CSV (timestamp_ms, raw_adc, voltage_mV) i emitujący je przez UART 921 600 baud." },
        { label: "Lokalizacja:", value: "HCMP_001 ESP32 LilyGO T-Display" },
        { label: "Powiązania:", value: "SCMP_004 radar_viewer.py" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("SCMP_003", "script.py – potok DSP dla IMU", [
        { label: "Opis:", value: "Skrypt Python: odczyt portu szeregowego, dekodowanie ramek IMU, PCA, filtry Butterwortha (pasmo oddechowe i sercowe), detekcja pików, zapis CSV, wykresy." },
        { label: "Lokalizacja:", value: "HCMP_004 Komputer PC badacza" },
        { label: "Powiązania:", value: "SCMP_001 Firmware IMU, SCMP_005 Skrypt ML" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("SCMP_004", "radar_viewer.py – akwizycja i DSP radaru", [
        { label: "Opis:", value: "Skrypt Python: podłączenie do ESP32 (921 600 baud), buforowanie próbek napięcia, live FFT (okno Hanninga), wizualizacja w czasie rzeczywistym (matplotlib), zapis CSV po sesji." },
        { label: "Lokalizacja:", value: "HCMP_004 Komputer PC badacza" },
        { label: "Powiązania:", value: "SCMP_002 Firmware ADC, SCMP_005 Skrypt ML" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("SCMP_005", "Skrypt trenowania modeli ML (CNN/LSTM)", [
        { label: "Opis:", value: "Moduł Python do przygotowania datasetu (segmentacja, labeling, normalizacja) i trenowania / ewaluacji modeli sieci neuronowych (CNN, LSTM) klasyfikujących wzorce oddechowe." },
        { label: "Lokalizacja:", value: "HCMP_004 Komputer PC badacza" },
        { label: "Powiązania:", value: "SCMP_003 script.py, SCMP_004 radar_viewer.py" },
        { label: "Źródło:", value: "STKH_002 Promotor pracy dyplomowej" },
        { label: "Priorytet:", value: "wysoki" },
    ]),
    sp(),
    br(),

    // ══════════════════════════════════════════════════════════════════════
    // 6. WYMAGANIA FUNKCJONALNE
    // ══════════════════════════════════════════════════════════════════════
    h1("6. Wymagania funkcjonalne"),
    body("Każde wymaganie funkcjonalne musi być przypisane do użytkownika lub systemu zewnętrznego (pole „Dotyczy”). Powinno wspierać cel biznesowy lub funkcjonalny bezpośrednio lub pośrednio. Wymagania mogą być grupowane wg użytkowników, zadań lub komponentów."),
  sp(),

    record("FNRQ_001", "Nawiązanie połączenia z ESP32 przez port szeregowy", [
        { label: "Opis:", value: "System musi wykrywać dostępne porty COM, próbować połączenia z priorytetem dla COM6 i poinformować użytkownika o wyniku w ciągu 5 s." },
        { label: "Dotyczy:", value: "USER_001 Badacz / Operator systemu pomiarowego" },
        { label: "Wsparcie dla:", value: "FNGL_001 Akwizycja IMU, FNGL_002 Akwizycja radaru" },
        { label: "Powiązania:", value: "FNRQ_002 Odbiór ramek IMU, FNRQ_003 Odbiór ramek radaru" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("FNRQ_002", "Odbiór i dekodowanie ramek danych IMU", [
        { label: "Opis:", value: "System musi odbierać linie CSV z UART (format: timestamp_ms, ax, ay, az, gx, gy, gz), walidować i buforować je w czasie rzeczywistym. Błędne linie są pomijane bez zatrzymania akwizycji." },
        { label: "Dotyczy:", value: "USER_001 Badacz / Operator systemu pomiarowego" },
        { label: "Wsparcie dla:", value: "FNGL_001 Akwizycja IMU" },
        { label: "Powiązania:", value: "FNRQ_004 DSP IMU, DTRQ_001 Surowe dane IMU" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("FNRQ_003", "Odbiór i dekodowanie ramek danych radaru", [
        { label: "Opis:", value: "System musi odbierać linie CSV z UART 921 600 baud (format: timestamp_ms, raw_adc, voltage_mV), walidować i buforować je. Błędne linie są pomijane." },
        { label: "Dotyczy:", value: "USER_001 Badacz / Operator systemu pomiarowego" },
        { label: "Wsparcie dla:", value: "FNGL_002 Akwizycja radaru" },
        { label: "Powiązania:", value: "FNRQ_005 DSP radaru, DTRQ_002 Surowe dane radaru" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("FNRQ_004", "Potok DSP dla sygnału IMU (PCA + Butterworth + FFT)", [
        { label: "Opis:", value: "System musi stosować: PCA (3 osie acc + 3 osie gyro) → filtry Butterwortha (oddech 0,1–0,5 Hz, tętno 0,65–4,0 Hz) → FFT/Welch → detekcja pików. Wyniki: BPM oddechu i tętna." },
        { label: "Dotyczy:", value: "USER_002 Analityk danych / Inżynier ML" },
        { label: "Wsparcie dla:", value: "FNGL_003 Przetwarzanie sygnałów" },
        { label: "Powiązania:", value: "FNRQ_002 Odbiór IMU, FNRQ_006 Zapis CSV, FNRQ_007 Wizualizacja" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("FNRQ_005", "Potok DSP dla sygnału radaru (Welch + FFT live)", [
        { label: "Opis:", value: "System musi detrendąć sygnał napięcia, stosować okno Hanninga, obliczać FFT z rozdzielczością 1024 próbek i estymować PSD metodą Welcha. Wynik: dominująca częstotliwość i prędkość Dopplera [m/s]." },
        { label: "Dotyczy:", value: "USER_002 Analityk danych / Inżynier ML" },
        { label: "Wsparcie dla:", value: "FNGL_003 Przetwarzanie sygnałów" },
        { label: "Powiązania:", value: "FNRQ_003 Odbiór radaru, FNRQ_006 Zapis CSV, FNRQ_007 Wizualizacja" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("FNRQ_006", "Zapis sesji pomiarowej do pliku CSV", [
        { label: "Opis:", value: "Po zakończeniu sesji system musi automatycznie zapisać wszystkie zebrane ramki do pliku CSV z nazwą zawierającą stempel czasowy (np. radar_raw_RRRR-MM-DD_HH-MM-SS.csv)." },
        { label: "Dotyczy:", value: "USER_001 Badacz / Operator systemu pomiarowego" },
        { label: "Wsparcie dla:", value: "FNGL_001 Akwizycja IMU, FNGL_002 Akwizycja radaru" },
        { label: "Powiązania:", value: "FNRQ_002 Odbiór IMU, FNRQ_003 Odbiór radaru, DTRQ_001, DTRQ_002" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("FNRQ_007", "Wizualizacja sygnału w czasie rzeczywistym", [
        { label: "Opis:", value: "Podczas sesji system musi wyświetlać wykres dziedziny czasu (ostatnie N próbek) i widmo FFT odnakścać dominującą częstotliwość z estymowanym statusem ruchu." },
        { label: "Dotyczy:", value: "USER_001 Badacz / Operator systemu pomiarowego" },
        { label: "Wsparcie dla:", value: "FNGL_005 Wizualizacja i raportowanie" },
        { label: "Powiązania:", value: "FNRQ_004 DSP IMU, FNRQ_005 DSP radaru" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "wysoki" },
    ]),
    sp(),

    record("FNRQ_008", "Przygotowanie datasetu do trenowania ML", [
        { label: "Opis:", value: "System musi umożliwiać segmentację sygnałów na okna czasowe, przypisanie etykiet klas (np. normalny oddech, bezdechy) i normalizację cech do zakresu [0,1]." },
        { label: "Dotyczy:", value: "USER_002 Analityk danych / Inżynier ML" },
        { label: "Wsparcie dla:", value: "FNGL_004 Budowa i trening modeli ML" },
        { label: "Powiązania:", value: "FNRQ_004 DSP IMU, FNRQ_005 DSP radaru, DTRQ_003 Dataset ML" },
        { label: "Źródło:", value: "STKH_002 Promotor pracy dyplomowej" },
        { label: "Priorytet:", value: "wysoki" },
    ]),
    sp(),

    record("FNRQ_009", "Trening i ewaluacja modeli CNN/LSTM", [
        { label: "Opis:", value: "System musi realizować trening modeli (CNN lub LSTM) na przygotowanym datasecie, obliczać metryki (accuracy, precision, recall, F1) na zbiorze testowym i zapisywać wytrenowany model." },
        { label: "Dotyczy:", value: "USER_002 Analityk danych / Inżynier ML" },
        { label: "Wsparcie dla:", value: "FNGL_004 Budowa i trening modeli ML" },
        { label: "Powiązania:", value: "FNRQ_008 Przygotowanie datasetu, DTRQ_003 Dataset ML" },
        { label: "Źródło:", value: "STKH_002 Promotor pracy dyplomowej" },
        { label: "Priorytet:", value: "wysoki" },
    ]),
    sp(),
    br(),

    // ══════════════════════════════════════════════════════════════════════
    // 7. WYMAGANIA NA DANE
    // ══════════════════════════════════════════════════════════════════════
    h1("7. Wymagania na dane"),
    body("Określają, jakie dane będą przetwarzane w systemie. Nie trzeba precyzować wszystkich szczegółów — te znajdą się w projekcie bazy danych."),
    sp(),

    record("DTRQ_001", "Surowe dane IMU (CSV 6-osiowy)", [
        { label: "Opis:", value: "Pliki CSV z polami: timestamp_ms (int), ax, ay, az (float, m/s²), gx, gy, gz (float, rad/s). Jedno nagranie = jeden plik z automatycznym stemplem w nazwie." },
        { label: "Powiązania:", value: "FNRQ_002 Odbiór IMU, FNRQ_004 DSP IMU, FNRQ_006 Zapis CSV" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("DTRQ_002", "Surowe dane radaru (CSV napięciowy)", [
        { label: "Opis:", value: "Pliki CSV z polami: timestamp_ms (int), raw_adc (int 0–4095), voltage_mV (float). Częstotliwość próbkowania ≥500 Hz; plik może przekraczać 100 MB dla długich sesji." },
        { label: "Powiązania:", value: "FNRQ_003 Odbiór radaru, FNRQ_005 DSP radaru, FNRQ_006 Zapis CSV" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("DTRQ_003", "Dataset ML – okna czasowe z etykietami", [
        { label: "Opis:", value: "Zbiór wycinków sygnału (tensory NumPy) z przypisanymi etykietami klas oddechowych. Format: .npy lub .h5. Zawiera podział train/val/test (np. 70/15/15%)." },
        { label: "Powiązania:", value: "FNRQ_008 Przygotowanie datasetu, FNRQ_009 Trening ML" },
        { label: "Źródło:", value: "STKH_002 Promotor pracy dyplomowej" },
        { label: "Priorytet:", value: "wysoki" },
    ]),
    sp(),
    br(),

    // ══════════════════════════════════════════════════════════════════════
    // 8. WYMAGANIA JAKOŚCIOWE
    // ══════════════════════════════════════════════════════════════════════
    h1("8. Wymagania jakościowe"),
    body("Wymagania jakościowe rozszerzają wymagania funkcjonalne. Powinny być z nimi powiązane. Podział odpowiada gałęziom drzewa jakości (za wyjątkiem funkcjonalności)."),
    sp(),

    h2("8.1. Wymagania w zakresie wiarygodności"),
    note("Dotyczą bezpieczeństwa, ochrony danych, odporności na błędy."),
    sp(120),
    record("RLRQ_001", "Anonimizacja danych osobowych ochotników", [
        { label: "Opis:", value: "Pliki CSV z danymi pomiarowymi nie mogą zawierać imienia, nazwiska ani innych danych identyfikujących. Nazwy plików zawierają tylko stempel czasowy." },
        { label: "Powiązania:", value: "FNRQ_006 Zapis CSV, RSRC_001 RODO" },
        { label: "Źródło:", value: "RSRC_001 RODO" },
        { label: "Priorytet:", value: "wysoki" },
    ]),
    sp(),

    record("RLRQ_002", "Odporność na błędne ramki UART", [
        { label: "Opis:", value: "Aplikacja PC musi kontynuować akwizycję po otrzymaniu uszkodzonej lub niepełnej linii CSV (np. z powodu zaklóceń USB). Błąd musi być logowany, nie powodować wyjątku." },
        { label: "Powiązania:", value: "FNRQ_002 Odbiór IMU, FNRQ_003 Odbiór radaru" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "wysoki" },
    ]),
    sp(),

    h2("8.2. Wymagania w zakresie wydajności"),
    note("Zastosowanie w czasie projektowania architektury systemu."),
    sp(120),
    record("PFRQ_001", "Minimalna częstotliwość próbkowania IMU", [
        { label: "Opis:", value: "Firmware ESP32 musi dostarczać ramki IMU z częstotliwością ≥200 Hz mierzoną na stronie PC. Wymaganie konieczne do poprawnej filtracji Butterwortha w pasmach oddechowym i sercowym." },
        { label: "Powiązania:", value: "FNRQ_002 Odbiór IMU, FNRQ_004 DSP IMU" },
        { label: "Źródło:", value: "RSRC_003 Karta katalogowa LSM6DS3" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("PFRQ_002", "Minimalna częstotliwość próbkowania ADC radaru", [
        { label: "Opis:", value: "Firmware ESP32 musi dostarczać ramki ADC z częstotliwością ≥500 Hz. Zapewnia wystarczające próbkowanie dla składowej oddechowej (do 0,6 Hz) zgodnie z twierdzeniem Nyquista z duży marginesem." },
        { label: "Powiązania:", value: "FNRQ_003 Odbiór radaru, FNRQ_005 DSP radaru" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    h2("8.3. Wymagania w zakresie elastyczności"),
    note("Zastosowanie w czasie wyboru koncepcji systemu (przenośność, skalowalność)."),
    sp(120),
    record("FLRQ_001", "Przenośność oprogramowania (Windows / Linux)", [
        { label: "Opis:", value: "Skrypty Python muszą działać bez modyfikacji na systemach Windows i Linux z Python ≥3.10. Zależności muszą być określone w pliku requirements.txt lub pyproject.toml." },
        { label: "Powiązania:", value: "SCMP_003 script.py, SCMP_004 radar_viewer.py" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "wysoki" },
    ]),
    sp(),

    h2("8.4. Wymagania w zakresie użyteczności"),
    note("Brane pod uwagę głównie w czasie projektowania interfejsu użytkownika."),
    sp(120),
    record("STRQ_001", "Komunikaty statusu w języku polskim", [
        { label: "Opis:", value: "Wszystkie komunikaty statusu wypisywane w konsoli i na wykresach muszą być w języku polskim (np. 'Nawiązywanie połączenia...', 'Status: RUCH WYKRYTY')." },
        { label: "Powiązania:", value: "FNRQ_001 Połączenie, FNRQ_007 Wizualizacja" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "średni" },
    ]),
    sp(),
    br(),

    // ══════════════════════════════════════════════════════════════════════
    // 9. SYTUACJE WYJĄTKOWE
    // ══════════════════════════════════════════════════════════════════════
    h1("9. Sytuacje wyjątkowe"),
    body("Każda sytuacja wyjątkowa musi być wspierana przez przynajmniej jedno wymaganie funkcjonalne, które rozwiązuje problem, zapobiega awarii lub pozwala odzyskać sprawność po awarii."),
    sp(),

    h2("9.1. Sytuacje nadzwyczajne"),
    note("Scenariusze rzadkie, ale możliwe — ich obsługa powinna być zautomatyzowana."),
    sp(120),
    record("EXCP_001", "Kliping sygnału wzmacniacza przy głębokim oddechu", [
        { label: "Opis:", value: "Przy dużych amplitudach klatki piersiowej sygnał IF po wzmocnieniu może przekroczyć 3,3 V i ulec klipingowi w ADC ESP32. Strata informacji o kształcie impulsów oddechowych." },
        { label: "Powiązania:", value: "FNRQ_003 Odbiór radaru, FNRQ_005 DSP radaru" },
        { label: "Wspierane przez:", value: "FNRQ_003 Odbiór ramek radaru (detekcja nasycenia ADC i ostrzeżenie)" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "wysoki" },
    ]),
    sp(),

    h2("9.2. Sytuacje krytyczne"),
    note("Scenariusze mogące doprowadzić do awarii — wymaganie wspierające zapobiega załamaniu systemu."),
    sp(120),
    record("CRIS_001", "Utrata połączenia USB podczas sesji pomiarowej", [
        { label: "Opis:", value: "Przypadkowe odłączenie przewodu USB w trakcie akwizycji powoduje utratę danych z bufora pamięci. Zagrożenie utraty całego nagrania." },
        { label: "Powiązania:", value: "FNRQ_001 Połączenie, FNRQ_006 Zapis CSV" },
        { label: "Wspierane przez:", value: "FNRQ_006 Zapis sesji do CSV (auto-flush co N sekund)" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "wysoki" },
    ]),
    sp(),

    h2("9.3. Sytuacje awaryjne"),
    note("Szkoda już nastąpiła — wymaganie wspierające pomaga odzyskać pełną sprawność systemu."),
    sp(120),
    record("EMRG_001", "Uszkodzenie pliku CSV po zerwaniu zasilania", [
        { label: "Opis:", value: "Awaryjne wyłączenie komputera lub ESP32 może spowodować nadpisanie częściowo zapisanego pliku CSV danymi zerowymi. Dane z sesji mogą być nieodwracalnie utracone." },
        { label: "Powiązania:", value: "FNRQ_006 Zapis CSV, DTRQ_001, DTRQ_002" },
        { label: "Wspierane przez:", value: "FNRQ_006 Zapis sesji do CSV (tryb 'append' zamiast nadpisywania)" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "średni" },
    ]),
    sp(),
    br(),

    // ══════════════════════════════════════════════════════════════════════
    // 10. DODATKOWE WYMAGANIA
    // ══════════════════════════════════════════════════════════════════════
    h1("10. Dodatkowe wymagania"),
    body("Wymagania nieujęte w poprzednich kategoriach."),
    sp(),

    h2("10.1. Wymagania sprzętowe"),
    record("XHRQ_001", "Wymagania sprzętowe dla stanowiska pomiarowego", [
        { label: "Opis:", value: "Stanowisko musi składać się z: ESP32 LilyGO T-Display, modułu LSM6DS3, modułu HB100 z wzmacniaczem MCP6002, kabla USB-C, zasilacza 5V/1A, komputera PC z USB." },
        { label: "Dotyczy:", value: "HCMP_001 ESP32, HCMP_002 LSM6DS3, HCMP_003 HB100" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    h2("10.2. Wymagania programowe"),
    note("Potrzeby klienta dotyczące oprogramowania (nie decyzje projektowe zespołu)."),
    sp(120),
    record("XSRQ_001", "Python ≥3.10 z bibliotekami naukowymi", [
        { label: "Opis:", value: "środowisko PC musi mieć zainstalowaną Pythona ≥3.10 oraz biblioteki: pyserial, numpy, scipy, pandas, matplotlib, PyTorch (lub TensorFlow) w wersjach kompatybilnych z CUDA (opcja GPU)." },
        { label: "Dotyczy:", value: "SSYS_003 Podsystem DSP, SSYS_004 Podsystem ML" },
        { label: "Źródło:", value: "STKH_003 Badacz / Autor systemu" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    h2("10.3. Inne wymagania"),
    record("XXRQ_001", "Zgłaszanie postepów do promotora", [
        { label: "Opis:", value: "Wyniki kolejnych etapów (dane IMU, wyniki radaru, metryki ML) muszą być dokumentowane i prezentowane promotorowi zgodnie z harmonogramem (do 20 czerwca 2026 – etap testowy, wrzesień–październik 2026 – ML)." },
        { label: "Dotyczy:", value: "STKH_002 Promotor pracy dyplomowej" },
        { label: "Źródło:", value: "STKH_002 Promotor pracy dyplomowej" },
        { label: "Priorytet:", value: "wysoki" },
    ]),
    sp(),
    br(),

    // ══════════════════════════════════════════════════════════════════════
    // 11. KRYTERIA AKCEPTACYJNE
    // ══════════════════════════════════════════════════════════════════════
    h1("11. Kryteria akceptacyjne"),
    body("Kryteria, jakim zostanie poddany gotowy system przed ostatecznym przyjęciem. Określa się warunki przeprowadzania testów lub okres próbnej eksploatacji. Nie podaje się oczywistych kryteriów jak spełnienie wymagań o priorytecie krytycznym."),
    sp(),

    record("ACPT_001", "Poprawna ekstrakcja rytmu oddechowego z IMU", [
        { label: "Opis:", value: "System musi poprawnie wyodrębniać częstotliwość oddechową w pasmie 0,1–0,6 Hz z dokładnością ±0,05 Hz w porównaniu z referencją (np. taśmą oddechową) dla co najmniej 80% sesji testowych." },
        { label: "Dotyczy:", value: "FNRQ_004 DSP IMU, BSGL_001 Nieinwazyjny monitoring" },
        { label: "Źródło:", value: "STKH_002 Promotor pracy dyplomowej" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("ACPT_002", "Detekcja oddechu przez radar HB100 na odległości do 50 cm", [
        { label: "Opis:", value: "Radar musi rejestrować wyraźny pik w widmie Dopplera odpowiadający oddechowi badanego przy odległości ≤50 cm w warunkach laboratoryjnych." },
        { label: "Dotyczy:", value: "FNRQ_005 DSP radaru, BSGL_001 Nieinwazyjny monitoring" },
        { label: "Źródło:", value: "STKH_002 Promotor pracy dyplomowej" },
        { label: "Priorytet:", value: "krytyczny" },
    ]),
    sp(),

    record("ACPT_003", "Dokładność modelu ML ≥80%", [
        { label: "Opis:", value: "Wytrenowany model CNN lub LSTM musi osiągać accuracy ≥80% i F1 ≥75% na zbiorze testowym przy klasyfikacji wzorców oddechowych." },
        { label: "Dotyczy:", value: "FNRQ_009 Trening ML, BSGL_002 Wkład naukowy" },
        { label: "Źródło:", value: "STKH_002 Promotor pracy dyplomowej" },
        { label: "Priorytet:", value: "wysoki" },
    ]),
    sp(),

    // ══════════════════════════════════════════════════════════════════════
    // 12. SŁOWNIK
    // ══════════════════════════════════════════════════════════════════════
    h1("12. Słownik"),
    body("IMU (Inertial Measurement Unit) – układ pomiaru bezwładnościowego zawierający akcelerometr i żyroskop."),
    body("ADC (Analog-to-Digital Converter) – przetwornik analogowo-cyfrowy (12-bitowy w ESP32)."),
    body("PCA (Principal Component Analysis) – analiza składowych głównych, metoda redukcji wymiarowości i uniezależnienia pomiarów od orientacji czujnika."),
    body("FFT (Fast Fourier Transform) – szybka transformata Fouriera, algorytm przeliczenia sygnału dziedziny czasu na dziedzinę częstotliwości."),
    body("PSD (Power Spectral Density) – gęstość widmowa mocy, estymowana metodą Welcha."),
    body("DSP (Digital Signal Processing) – cyfrowe przetwarzanie sygnałów."),
    body("CNN (Convolutional Neural Network) – splotowa sieć neuronowa."),
    body("LSTM (Long Short-Term Memory) – rekurencyjna sieć neuronowa z długoterminową pamięcią, stosowana do analizy szeregów czasowych."),
    body("Doppler (efekt) – zmiana obserwowanej częstotliwości fali mikrofalowej wynikająca z ruchu obiektu względem źródła. Radar HB100 mierzy przemieszczenie klatki piersiowej jako przesuniecie częstotliwości."),
    body("BPM (Beats Per Minute) – jednostka częstotliwości rytmu biologicznego (odd. lub serca) w minutach."),
    body("ESP-IDF – Espressif IoT Development Framework, oficjalne środowisko programowania firmware dla ESP32."),
    sp(),

    // ══════════════════════════════════════════════════════════════════════
    // 13. ZAŁĄCZNIKI
    // ══════════════════════════════════════════════════════════════════════
    h1("13. Załączniki"),
    body("1. Schemat elektryczny układu wzmacniacza MCP6002 dla radaru HB100 (KiCad, plik: esp32_imu.kicad_sch)."),
    body("2. Przykładowe nagrania CSV z sesji IMU (katalog: respiratory_6axis_raw_*.csv)."),
    body("3. Przykładowe nagrania CSV z sesji radaru (katalog: radar_raw_*.csv)."),
    body("4. Prezentacja seminarium dyplomowego inżynierskiego (plik: prezentacja.tex / seminarium-2.pdf)."),
    body("5. Karta katalogowa LSM6DS3 – STMicroelectronics (URL: https://www.st.com/en/mems-and-sensors/lsm6ds3.html)."),
    body("6. Karta katalogowa HB100 (URL: https://www.icstation.com/hb100-doppler-microwave-module.html)."),
    sp(),
);

// ══════════════════════════════════════════════════════════════════════════════
//  BUILD
// ══════════════════════════════════════════════════════════════════════════════
const doc = new Document({
    styles: {
        default: { document: { run: { font: FONT, size: 22 } } },
        paragraphStyles: [
            {
                id: "Heading1", name: "Heading 1",
                basedOn: "Normal", next: "Normal", quickFormat: true,
                run: { font: FONT, size: 28, bold: true, allCaps: true },
                paragraph: {
                    spacing: { before: 480, after: 240 }, outlineLevel: 0,
                    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "000000", space: 4 } },
                },
            },
            {
                id: "Heading2", name: "Heading 2",
                basedOn: "Normal", next: "Normal", quickFormat: true,
                run: { font: FONT, size: 24, bold: true },
                paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 1 },
            },
            {
                id: "Heading3", name: "Heading 3",
                basedOn: "Normal", next: "Normal", quickFormat: true,
                run: { font: FONT, size: 22, bold: true, italics: true },
                paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 2 },
            },
        ],
    },
    sections: [{
        properties: {
            page: {
                size: { width: PAGE_W, height: 16838 },
                margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
            },
        },
        headers: {
            default: new Header({
                children: [new Paragraph({
                    spacing: { after: 100 },
                    border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: "000000", space: 4 } },
                    children: [new TextRun({ text: "Specyfikacja wymagań systemowych", font: FONT, size: 18, italics: true })],
                })]
            }),
        },
        footers: {
            default: new Footer({
                children: [new Paragraph({
                    alignment: AlignmentType.CENTER,
                    border: { top: { style: BorderStyle.SINGLE, size: 2, color: "000000", space: 4 } },
                    children: [
                        new TextRun({ text: "Strona ", font: FONT, size: 18 }),
                        new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 }),
                        new TextRun({ text: " / ", font: FONT, size: 18 }),
                        new TextRun({ children: [PageNumber.TOTAL_PAGES], font: FONT, size: 18 }),
                    ],
                })]
            },),
        },
        children: C,
    }],
});

const OUT = "./SiecINeuronowe-RytmOddechowy-1.docx";
Packer.toBuffer(doc).then(buf => {
    fs.writeFileSync(OUT, buf);
    console.log("Saved:", OUT);
}).catch(err => { console.error(err); process.exit(1); });