# Plan jednodniowych pomiarów przed spotkaniem z promotorem

Data przygotowania: 15.07.2026

## Cel dnia

W ciągu jednego dnia zebrać mały, ale metodologicznie czytelny zestaw danych,
który pozwoli odpowiedzieć na cztery pytania:

1. Czy sygnał oddechowy z aplikacji iPhone i z LSM6DS3 jest użyteczny w
   pozycji siedzącej i leżącej?
2. Czy HB100 i A121 mogą pracować jednocześnie bez widocznego wzajemnego
   zakłócania?
3. Gdzie w praktyce kończy się wykrywanie oddechu przez A121 i HB100 przy
   niezmienionej metodzie analizy?
4. Czy folia daje jakąkolwiek korzyść przy 2 m i czy wynik zależy od tego,
   czy radar pada na klatkę pod naturalnym kątem, czy niemal prostopadle?

Tego dnia nie powstaje szyna ani fantom z silnikiem. Część z siecią neuronową
zostaje do ustalenia z promotorem.

## Zasady wspólne

- Jedna osoba, ta sama odzież, możliwie to samo położenie czujników i stałe
  ustawienie fotela/łóżka w obrębie serii.
- Oddech zadany: metronom 12 odd./min, czyli 0,20 Hz. Wstrzymanie oddechu jest
  kontrolą negatywną, ale nie jest wzorcem amplitudy ani ruchu serca.
- Nagranie podstawowe trwa 60 s. Trzydzieści sekund daje przy 12 odd./min tylko
  sześć cykli, dlatego nadaje się do szybkiej kontroli lub bezdechu, ale jest
  słabe do stabilnej oceny widma.
- Schemat nagrania 60 s: 0–10 s uspokojenie, 10–45 s warunek właściwy,
  45–60 s wstrzymanie oddechu. Jeżeli wstrzymanie 15 s jest niekomfortowe,
  skrócić je; nie forsować go.
- Warunki A/B wykonywać parami i odwracać kolejność, np. AB, BA, AB. Nie robić
  całego bloku A, a potem całego bloku B.
- Nie używać `resp_confidence` ani `signal_quality` jako głównych wyników,
  ponieważ w dotychczasowych danych nasycały się. Zapisywać SNR widmowy,
  prominencję piku, trafienie zadanej częstości i stabilność między oknami.
- Przed startem utworzyć dziennik sesji. Minimalne kolumny:
  `run_id, czas, urządzenia, pozycja, dystans, folia, konfiguracja, kolejność,
  pliki, uwagi`.
- Po każdej serii od razu wpisać nazwy powstałych plików. Pliki HB100 nie mają
  etykiety warunku w nazwie, więc bez dziennika szybko tracą znaczenie.
- Dystans ustawiać tym samym dalmierzem MILESEEY D2 40 m. Radar pozostawić na
  statywie Manfrotto, a plamką lasera sprawdzać orientacyjne skierowanie osi na
  środek klatki piersiowej. W dzienniku zapisywać także wysokość i kąt albo
  przesunięcie poprzeczne stanowiska.

## Warunek konieczny: czas LSM6DS3

Obecny firmware LSM6DS3 wysyła sześć wartości bez czasu próbki, a komputer
nadaje czas przy odbiorze linii UART. Buforowanie może tworzyć zduplikowane lub
silnie skupione timestampy. Przed analizą ilościową trzeba:

1. dodać w firmware czas z `esp_timer_get_time()` albo monotonny licznik
   próbki,
2. przesyłać go jako pierwsze pole wiersza,
3. odczytać go w `src/respi_net/imu.py`, zamiast nadawać każdej linii
   `time.time()` na hoście,
4. w 60-sekundowym nagraniu nieruchomego czujnika sprawdzić medianę, histogram
   i percentyle odstępów czasu oraz liczbę odstępów równych zero.

Jeśli poprawka nie działa do 09:30, nie blokować całego dnia. Wykonać pełny
test iPhone, a LSM6DS3 potraktować wyłącznie jakościowo i jawnie opisać
ograniczenie czasowe.

## Harmonogram

| Czas | Zadanie | Wynik |
|---|---|---|
| 08:30–09:30 | timestamp LSM6DS3, test 60 s obu urządzeń nieruchomo | poprawna oś czasu lub jawny fallback |
| 09:30–10:45 | iPhone + LSM6DS3: siedząc i leżąc | 8 wspólnych nagrań po 60 s |
| 10:45–11:30 | test zakłóceń HB100–A121 | decyzja: razem czy osobno |
| 11:30–13:00 | przerwa i przygotowanie stanowiska dystansowego | geometria i lista dystansów |
| 13:00–14:30 | zasięg A121; równolegle HB100, jeśli test zakłóceń przeszedł | krzywa trafień/SNR względem dystansu |
| 14:30–15:45 | retest folii przy 2 m: geometria naturalna i prostopadła, tylko soczewka hiperboliczna | układ 2×2: kąt × folia |
| 15:45–16:30 | opcjonalny test odchylenia od osi A121, jeżeli wcześniejsze bloki skończą się poprawnie | sprawdzenie hipotezy o szerokości pola widzenia |
| 16:30–18:00 | kontrola plików, analiza, wykresy, kopia danych | materiał do pracy i dla promotora |

## 1. Test iPhone oraz LSM6DS3

Umieścić telefon i LSM6DS3 obok siebie na górnej części tułowia. Nie są w tym
eksperymencie referencją dla radaru; porównywane są dwa warianty samodzielnej
metody inercyjnej. Pozycja nie będzie identyczna co do milimetra, więc
porównanie amplitud między urządzeniami ma znaczenie drugorzędne.

Macierz minimalna:

| ID | Pozycja | Oddech | Powtórzenia | Czas |
|---|---|---:|---:|---:|
| IMU-S-N | siedząca | swobodny + końcowe wstrzymanie | 2 | 60 s |
| IMU-S-12 | siedząca | 12/min + końcowe wstrzymanie | 2 | 60 s |
| IMU-L-N | na plecach | swobodny + końcowe wstrzymanie | 2 | 60 s |
| IMU-L-12 | na plecach | 12/min + końcowe wstrzymanie | 2 | 60 s |

Łącznie: 8 minut czystego nagrania. Nie rozszerzać tego dnia macierzy o bok,
brzuch, stanie i wiele lokalizacji — te warianty są dobre na później, jeżeli
podstawowy sygnał okaże się użyteczny. Zestaw siedzenie/leżenie jest
wystarczający, jeżeli w pracy wniosek ograniczy się do użyteczności sygnału
w tych dwóch pozycjach. Dodatkowe nachylenia byłyby potrzebne dopiero do
obrony mocniejszej tezy, że PCA uniezależnia wynik od dowolnej orientacji.

Komendy uruchamiane w dwóch terminalach:

```bash
uv run respi capture-imu --port <PORT_LSM6> --no-plot
uv run respi capture-iphone-imu --seconds 60 --no-plot
```

LSM6DS3 kończy się `Ctrl+C`; warto uruchomić go kilka sekund przed iPhone'em i
zatrzymać kilka sekund po nim.

Raportować osobno dla przyspieszenia i żyroskopu/PCA:

- rzeczywistą częstość próbkowania i histogram `Δt`,
- liczbę braków/powtórzeń czasu; dla BLE także luki sekwencji paczek (obecny
  odbiornik zachowuje ostatni numer, ale nie zlicza jeszcze przeskoków),
- częstotliwość dominującą i błąd względem 0,20 Hz w próbach z metronomem,
- SNR piku oddechowego i jego spadek podczas wstrzymania oddechu,
- stabilność wyniku w pozycji siedzącej i leżącej,
- udział wariancji wyjaśniony przez pierwszą składową PCA.

Robocze kryterium użyteczności: pik przy 0,20 Hz z błędem nie większym niż
0,02 Hz, co najmniej 6 dB nad medianą tła w paśmie oraz wyraźny spadek mocy
oddechowej podczas wstrzymania. Progi są kryteriami tego eksperymentu, a nie
normą kliniczną.

## 2. Czy HB100 i A121 mogą nagrywać równocześnie?

Ich częstotliwości podstawowe są daleko od siebie: HB100 pracuje przy
10,525 GHz, a A121 w paśmie 57–64 GHz. Bezpośrednie zakłócenie na
częstotliwości podstawowej jest więc mało prawdopodobne. Nie wolno jednak
przyjąć z góry, że zakłóceń nie ma: szósta harmoniczna HB100 wypada przy około
63,15 GHz, czyli wewnątrz pasma A121, a zakłócenie może też wejść przez wspólne
zasilanie USB lub przewody.

Ustawić oba radary tak jak w pomiarze docelowym: obok siebie, w odległości
około 10–20 cm, równolegle. Wykonać po dwa powtórzenia:

| Warunek | A121 | HB100 | Scena |
|---|---|---|---|
| INT-A | nagrywa | wyłączony z zasilania | pusty, nieruchomy pokój |
| INT-B | nagrywa | włączony i nagrywa | pusty, nieruchomy pokój |
| INT-C | wyłączony z zasilania | nagrywa | pusty, nieruchomy pokój |

Każde nagranie: 60 s. Dla A121 porównać amplitudę tła i widmo szumu fazowego w
tej samej bramce; dla HB100 — RMS i widmo napięcia w pasmach 0,1–0,6 Hz oraz
0,6–5 Hz. Zmiana większa niż 3 dB, nowa stała linia widmowa albo wyraźna utrata
ramek jest sygnałem ostrzegawczym. Wtedy powtórzyć jedną parę z osobnymi
zasilaniami/hubami. Jeśli problem pozostaje, nie nagrywać radarów równocześnie.

Przykładowe terminale dla warunku wspólnego:

```bash
uv run respi capture-radar --port <PORT_HB100> --no-plot
uv run respi record-a121-test --port <PORT_A121> --seconds 60 \
  --label int-b-both --start-m 0.2 --end-m 1.5 \
  --profile 3 --hwaas 32 --sweeps-per-frame 8 --frame-rate-hz 20
```

## 3. Zasięg i praktyczny limit HB100/A121

### Wariant minimalny do wykonania teraz

Jeżeli celem jest przed spotkaniem pokazać przede wszystkim, **co da się
wycisnąć z HB100**, nie trzeba od razu wykonywać pełnej serii do 4 m. Najkrótszy
zestaw, który zachowuje sens metodyczny, wygląda następująco:

1. Zamocować HB100 i A121 obok siebie, z odległością środków apertur nie większą
   niż około 5–10 cm, na wysokości mostka i z osiami możliwie równoległymi.
   A121 pozostawić z płaską pokrywą: w tym badaniu jest równoległym torem
   porównawczym, a nie przedmiotem kolejnego testu soczewek.
2. W pustej, nieruchomej scenie wykonać po 30 s: A121 przy wyłączonym HB100,
   oba radary włączone oraz HB100 przy wyłączonym A121. Jest to tylko szybki
   test przesiewowy zakłóceń. Gdy pojawi się nowa linia widmowa, wzrost tła
   ponad 3 dB albo utrata ramek, podejrzaną parę powtórzyć z osobnymi portami
   USB/zasilaniem; przy braku różnicy przejść od razu do pomiarów wspólnych.
3. Wykonać wspólne nagrania przy 30 cm (jedno nagranie kontrolne), 60 cm (dwa
   powtórzenia) i 100 cm (dwa powtórzenia). Nie zmieniać wzmocnienia toru HB100
   między dystansami i zapisać, czy ADC nie ulega nasyceniu.
4. Jeżeli 100 cm spełnia kryterium użyteczności, wykonać jedno nagranie przy
   150 cm. Po sukcesie zwiększać dystans co 50 cm aż do pierwszego
   niepowodzenia. Powtórzyć tylko najdalszy dystans zaliczony oraz pierwszy
   niezaliczony; to wystarcza do podania przedziału praktycznej granicy bez
   rozbudowanej macierzy.

Każde nagranie powinno trwać 90 s i powtarzać już użyty protokół A121:
10 s oddechu swobodnego, 7 cykli po 5 s (wdech 2 s, wydech 3 s), 15 s
wstrzymania po wydechu oraz 6 kolejnych cykli. Na początku wspólnego zapisu
wykonać trzy krótkie ruchy tułowia do późniejszego wyrównania osi czasu.
Warunek zalicza się dla HB100, gdy pik przy 0,20 ± 0,02 Hz jest obecny w obu
częściach oddychania, ma co najmniej 6 dB nad tłem i wyraźnie zanika podczas
wstrzymania. A121 służy do sprawdzenia, czy w tym samym czasie rzeczywiście
zarejestrowano zadany przebieg; nie jest wzorcem klinicznym.

Taki wariant zajmuje około 10–15 minut czystego nagrywania plus zmiany
dystansu. Daje najlepszy przypadek, dwa powtórzenia przy odległościach
praktycznych oraz przedział granicy zasięgu. Dodatkowe dystanse i pełne
odwrócenie kolejności mają sens dopiero wtedy, gdy ta seria działa i ma zostać
użyta do mocniejszego wniosku ilościowego.

### Wariant rozszerzony z pełną krzywą dystansu

Po pozytywnym teście zakłóceń nagrywać oba radary równocześnie. W tym
rozszerzonym wariancie A121 ma tylko soczewkę hiperboliczną; soczewki
zaprojektowanej dla około 60 GHz nie należy
zakładać na HB100 pracujący przy 10,525 GHz. Radary ustawić na wysokości
mostka, równolegle, bez zmiany statywów w całej serii.

Kolejność pierwszej serii, celowo niemonotoniczna:

`0,5 m → 2,0 m → 1,0 m → 3,0 m → 1,5 m → 4,0 m → 2,5 m`

Jeżeli pomieszczenie nie pozwala na 4 m, zakończyć na najdalszym mierzalnym
dystansie i podać ograniczenie pomieszczenia. Zrobić dwa powtórzenia każdego
dystansu, drugie w odwrotnej kolejności. Pomiędzy powtórzeniami wstać i usiąść
ponownie — pokaże to praktyczną wrażliwość HB100 na małą zmianę średniej
odległości bez budowania precyzyjnej szyny.

Stałe ustawienia A121: profil 3, HWAAS 32, 8 sweepów/ramkę, 20 Hz. Okno
odległościowe można przesuwać wraz z celem, ale powinno mieć stałą szerokość
około 1 m. Przykład dla 2 m:

```bash
uv run respi record-a121-test --port <PORT_A121> --seconds 60 \
  --label range-2m-r1 --start-m 1.5 --end-m 2.5 \
  --profile 3 --hwaas 32 --sweeps-per-frame 8 --frame-rate-hz 20
```

W drugim terminalu podczas tego samego warunku:

```bash
uv run respi capture-radar --port <PORT_HB100> --no-plot
```

Na początku i końcu wspólnego pomiaru wykonać trzy krótkie, wyraźne ruchy
tułowiem. Znaczniki HB100 są czasem od uruchomienia ESP32, a A121 używa czasu
hosta, więc te ruchy pozwalają później wyrównać przebiegi. Do porównania samej
częstości oddechu dokładna synchronizacja nie jest konieczna.

Za trafienie uznać pik `0,20 ± 0,02 Hz`, obecny w kolejnych oknach i z SNR co
najmniej 6 dB. Dla HB100 dodatkowo policzyć stosunek energii 0,20 Hz do drugiej
harmonicznej 0,40 Hz. Dominacja 0,40 Hz jest praktycznym przykładem ograniczenia
jednokanałowego CW, a nie poprawną detekcją 24 odd./min.

Zasięg roboczy raportować jako najdalszy dystans spełniający kryterium w obu
powtórzeniach. Najbliższy dystans niespełniający kryterium powtórzyć trzy razy
po 60 s. Jeśli A121 zawiedzie, wykonać tylko jedną serię ratunkową z profilem 4
i HWAAS 128; oznaczyć ją jako konfigurację strojoną i nie mieszać z serią
podstawową.

## 4. Retest folii przy 2 m: geometria naturalna i prostopadła

**Status: wykonano 15 lipca 2026.** Zarejestrowano komplet 12 prób w sesji
`guided_foil2m_2026-07-15_16-26-15`. Analizę odtwarza
`tools/analyze_a121_foil_2m.py`; wyniki i wnioski wpisano do sekcji
„Retest folii i kąta padania na dystansie 2 m” pracy.

Celem retestu jest rozdzielenie dwóch możliwych wyjaśnień dotychczasowego
wyniku: folia rzeczywiście nie poprawia pomiaru albo przy naturalnym kącie
odbicie zwierciadlane jest kierowane poza radar.

Ustawienia wspólne:

- wyłącznie A121 z soczewką hiperboliczną;
- dystans 2,0 m, profil 3, HWAAS 32, 8 sweepów/ramkę, 20 Hz, zakres 1,5–2,5 m;
- pozycja siedząca na tym samym krześle i ta sama warstwa odzieży;
- łata 5 × 5 cm; jej górna krawędź 5 cm poniżej poziomej linii łączącej
  brodawki sutkowe;
- każde nagranie 90 s według identycznych komunikatów programu:
  0–10 s oddech swobodny, 10–45 s siedem cykli 12/min (wdech 2 s, wydech
  3 s), 45–60 s wstrzymanie po wydechu i 60–90 s sześć kolejnych cykli
  12/min. Wstrzymania nie forsować; przy dyskomforcie przerwać i odrzucić
  próbę.

Badane są dwa ustawienia geometryczne:

- **N — naturalne:** niewymuszona pozycja siedząca, radar równoległy do
  podłoża i sprawdzony poziomnicą pęcherzykową. Oś radaru tworzy około 30° z
  normalną do klatki, tak jak w pierwszym eksperymencie.
- **P — prawie prostopadłe:** wiązka pada możliwie pod kątem 90° do
  powierzchni klatki, czyli 0° względem jej normalnej. Najlepiej uzyskać to
  przez zmianę pochylenia i wysokości głowicy statywu, bez wymuszania innej
  pozycji ciała. Oś nadal ma trafiać w środek łaty, a odległość pozostać 2,0 m.

Dla każdego ustawienia wykonywane są nagrania z folią i bez folii:

| Kod | Geometria | Folia |
|---|---|---|
| N0 | naturalna, około 30° od normalnej | brak |
| NF | naturalna, około 30° od normalnej | jest |
| P0 | prawie 90° do powierzchni / 0° od normalnej | brak |
| PF | prawie 90° do powierzchni / 0° od normalnej | jest |

### Weryfikacja ustawienia prostopadłego laserem

1. Dalmierz ustawić sztywno przy radarze tak, aby jego oś lasera była
   równoległa do osi A121; sposób mocowania sfotografować.
2. Z założoną i możliwie wygładzoną folią skierować laser w środek łaty.
3. Regulować pochylenie głowicy statywu do chwili, gdy odbity punkt lasera
   wraca możliwie blisko miejsca emisji. Powrót obserwować na małej matowej,
   białej kartce umieszczonej przy aperturze dalmierza, a nie patrząc wzdłuż
   wiązki ani w jej zwierciadlane odbicie.
4. Zablokować głowicę i zaznaczyć pozycję krzesła. Ustawienie opisać jako
   „prawie prostopadłe”, ponieważ pofałdowana folia i krzywizna klatki nie
   tworzą idealnej płaszczyzny.
5. Zaznaczyć dwa powtarzalne położenia głowicy statywu: N (naturalne) i P
   (prawie prostopadłe). W bloku z folią przełączać wyłącznie położenie
   statywu, nie odklejając łaty.
6. Po szóstej próbie, zakończonej w warunku PF, zdjąć folię jeden raz bez
   zmiany statywu ani krzesła i od razu wykonać P0. W całym drugim bloku
   odtwarzać N i P z oznaczeń, bez ponownego przyklejania folii.

Każdy z czterech warunków wykonać trzy razy. Folia jest przyklejana tylko raz
przed próbą 1 i zdejmowana tylko raz między próbą 6 a 7. Kolejność geometrii
jest odwrócona w drugim bloku:

- blok z folią: `NF → PF → PF → NF → NF → PF`,
- blok bez folii: `P0 → N0 → N0 → P0 → P0 → N0`.

Taki układ eliminuje odklejanie i ponowne przyklejanie łaty, ale nie rozdziela
w pełni efektu folii od dryfu w czasie: wszystkie pomiary z folią są wykonane
wcześniej. Wynik należy więc opisać jako pilotażowy, a spójny spadek lub wzrost
wszystkich metryk w drugim bloku traktować również jako możliwy efekt sesji.

Daje to 12 nagrań i po trzy realizacje każdego warunku. Przykładowa komenda:

```bash
uv run python tools/a121_foil_2m_experiment.py
```

Runner sam realizuje powyższą kolejność i wyraźnie wskazuje moment jednorazowego
zdjęcia folii. Pokazuje duże komunikaty `WDECH`,
`WYDECH` i `WSTRZYMAJ ODDECH`, odlicza czas do zmiany oraz zapisuje rozwiniętą
oś poleceń w manifeście sesji. `Esc` lub czerwony przycisk przerywa aktywną
próbę i usuwa jej plik; ten sam warunek pozostaje wtedy gotowy do ponowienia.

Najpierw policzyć efekt folii osobno dla obu geometrii:

- `ΔN = NF − N0` — efekt folii w pozycji naturalnej,
- `ΔP = PF − P0` — efekt folii przy padaniu prawie prostopadłym,
- `ΔP − ΔN` — informacja, czy korzyść z folii zależy od kąta.

Porównać amplitudę w bramce klatki, SNR, prominencję piku, stabilność
odległości i trafienie 0,20 ± 0,02 Hz. Jeśli poprawa pojawi się tylko w PF,
wynik będzie wspierał hipotezę o kierowaniu odbicia poza radar w naturalnej
geometrii. Brak poprawy w NF i PF będzie wskazywał, że łata nie daje praktycznej
korzyści również przy korzystnym ustawieniu. Przy trzech powtórzeniach wynik
pozostaje pilotażowy: pokazać wszystkie punkty i mediany różnic, bez mocnego
wniosku opartego wyłącznie na p-value.

## 5. Opcjonalny test odchylenia od osi A121

Ten test ma większą wartość dla głównego, radarowego tematu pracy niż dalsze
rozszerzanie macierzy IMU. Sprawdza hipotezę, że soczewka skupiająca daje
większy zysk przy poprawnym wycelowaniu, ale płaska pokrywa jest bardziej
odporna na przesunięcie śpiącej osoby poza oś radaru.

- Dystans nominalny: 1,0 m. Przy tym dystansie dotychczasowy SNR jest duży,
  więc spadku jakości nie powinien powodować sam brak mocy.
- Warianty: soczewka hiperboliczna i płaska pokrywa; FZP pominąć.
- Położenie środka klatki poza osią głównej wiązki: 0°, 15° i 30°.
- Dwa powtórzenia każdego warunku, po 60 s, oddech 12/min.
- Kolejność przeplatać między soczewkami i kątami.
- Metryki: amplituda w bramce klatki, SNR, prominencja piku, stabilność
  śledzonej odległości i odsetek okien trafiających 0,20 ± 0,02 Hz.

Jeżeli statyw nie ma wiarygodnej podziałki kąta, można pozostawić radar
nieruchomo i przesunąć badanego poprzecznie względem osi. Dla odległości 1 m
przesunięcia odpowiadające w przybliżeniu 15° i 30° wynoszą odpowiednio
0,27 m i 0,58 m (`x = R·tan(θ)`). Bez wykonania tego testu w pracy należy
pozostawić sformułowanie „prawdopodobne wyjaśnienie geometryczne”, a nie
„przyczyną była niewłaściwa pozycja radaru”.

Nie należy mylić tego testu z kątem padania na powierzchnię klatki. Retest
z sekcji 4 porównuje 30° i 0° względem normalnej, aby zbadać kierunkowość
odbicia od folii. Test odchylenia od osi pozostawia orientację tułowia stałą,
ale przesuwa jego środek poza główną oś radaru, aby zbadać pole widzenia
soczewek.

## Wykresy do zrobienia tego samego dnia

1. iPhone i LSM6DS3: sygnał + PSD dla siedzenia i leżenia oraz histogram `Δt`.
2. Trafienie 0,20 Hz i SNR względem dystansu dla A121 i HB100.
3. HB100: energia składowej podstawowej 0,20 Hz oraz harmonicznej 0,40 Hz w
   każdym powtórzeniu.
4. Folia 2 m: delty `NF−N0` i `PF−P0` dla amplitudy i SNR oraz porównanie
   efektu folii między geometrią naturalną i prostopadłą.
5. Jeżeli wykonano test kąta: SNR i odsetek trafionych okien względem kąta,
   osobne krzywe dla soczewki hiperbolicznej i płaskiej pokrywy.

## Minimalny rezultat dla promotora

Nawet gdy część prób zawiedzie, dzień jest udany, jeżeli powstaną:

- jawny wniosek o jakości timestampów i BLE,
- tabela „siedząc/leżąc × iPhone/LSM6DS3” z kryteriami użyteczności,
- test wykluczający albo ujawniający zakłócenia HB100–A121,
- empiryczna granica odległości z warunkiem trafienia, a nie oceną „na oko”,
- uczciwy pilotaż folii przy 2 m wykonany w jednej przeplatanej sesji, osobno
  dla naturalnego kąta i padania prawie prostopadłego.

Test kąta jest dodatkiem o dużej wartości interpretacyjnej, ale nie jest
warunkiem minimalnego wyniku dnia.
