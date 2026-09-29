import AudioToolbox
import CoreBluetooth
import CoreMotion
import Foundation
import UIKit

private let imuServiceUUID = CBUUID(string: "7B61B4E2-F5B4-4C90-8C7F-A7B2F1E8F4D0")
private let imuDataUUID = CBUUID(string: "7B61B4E3-F5B4-4C90-8C7F-A7B2F1E8F4D0")
private let imuControlUUID = CBUUID(string: "7B61B4E4-F5B4-4C90-8C7F-A7B2F1E8F4D0")

// Preview of the breathing: gravity averaged to 10 points a second, last 30 s.
private let previewRateHz = 10.0
private let previewSeconds = 30.0

private struct MotionSample {
    let timeMs: UInt32
    let elapsedMs: Double
    let ax: Double
    let ay: Double
    let az: Double
    let gx: Double
    let gy: Double
    let gz: Double
}

/// One motion reading, from CoreMotion or (in the simulator) from a made-up breath.
private struct MotionReading {
    let timestamp: TimeInterval  // seconds since boot, like CMLogItem.timestamp
    let gravity: (x: Double, y: Double, z: Double)
    let acceleration: (x: Double, y: Double, z: Double)  // gravity + user acceleration, in g
    let rotationDegPerS: (x: Double, y: Double, z: Double)
}

/// Every streamed sample also goes to a CSV on the phone, so a Bluetooth dropout never loses a trial.
///
/// Same columns as the desktop IMU files (`Time_ms,ax,ay,az,gx,gy,gz`); `Time_ms` is Unix time in ms
/// from the phone's clock.  Files live in the app's Documents folder, which the Files app shows.
private final class BackupWriter {
    let url: URL
    private let queue = DispatchQueue(label: "RespiPhoneIMU.backup")
    private var handle: FileHandle?
    private var buffer = ""
    private var buffered = 0
    private let startUnixMs: Double

    init?(directory: URL, startUnixMs: Double) {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let name = "respi_imu_\(formatter.string(from: Date(timeIntervalSince1970: startUnixMs / 1000.0))).csv"
        url = directory.appendingPathComponent(name)
        guard FileManager.default.createFile(atPath: url.path, contents: Data("Time_ms,ax,ay,az,gx,gy,gz\n".utf8)),
              let handle = try? FileHandle(forWritingTo: url)
        else {
            return nil
        }
        handle.seekToEndOfFile()
        self.handle = handle
        self.startUnixMs = startUnixMs
    }

    func append(_ sample: MotionSample) {
        let line = String(
            format: "%.1f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f\n",
            startUnixMs + sample.elapsedMs, sample.ax, sample.ay, sample.az, sample.gx, sample.gy, sample.gz
        )
        queue.async {
            self.buffer += line
            self.buffered += 1
            if self.buffered >= 100 {
                self.flush()
            }
        }
    }

    func close() {
        queue.sync {
            flush()
            try? handle?.close()
            handle = nil
        }
    }

    private func flush() {
        guard let handle, !buffer.isEmpty else {
            return
        }
        handle.write(Data(buffer.utf8))
        buffer = ""
        buffered = 0
    }
}

final class MotionBLEStreamer: NSObject, ObservableObject {
    @Published var bluetoothState = "starting"
    @Published var isAdvertising = false
    @Published var isStreaming = false
    @Published var subscriberCount = 0
    @Published var sampleRateHz = UserDefaults.standard.object(forKey: "sampleRateHz") as? Double ?? 100.0 {
        didSet {
            UserDefaults.standard.set(sampleRateHz, forKey: "sampleRateHz")
            restartMotion()
        }
    }
    @Published var autoStartOnConnection = UserDefaults.standard.object(forKey: "autoStartOnConnection") as? Bool ?? false {
        didSet { UserDefaults.standard.set(autoStartOnConnection, forKey: "autoStartOnConnection") }
    }
    @Published var keepScreenAwake = UserDefaults.standard.object(forKey: "keepScreenAwake") as? Bool ?? true {
        didSet {
            UserDefaults.standard.set(keepScreenAwake, forKey: "keepScreenAwake")
            updateIdleTimer()
        }
    }
    @Published var dimScreenWhileStreaming = UserDefaults.standard.object(forKey: "dimScreenWhileStreaming") as? Bool ?? true {
        didSet { UserDefaults.standard.set(dimScreenWhileStreaming, forKey: "dimScreenWhileStreaming") }
    }
    @Published private(set) var samplesSent = 0
    @Published private(set) var batchesSent = 0
    @Published private(set) var latestBatchSize = 0
    @Published var statusMessage = "Waiting for Bluetooth."

    @Published private(set) var trace: [Double] = []  // breathing preview in mg, 10 points a second
    @Published private(set) var traceSpanMg = 0.0  // 5th-95th percentile spread of the preview
    @Published private(set) var measuredRateHz = 0.0
    @Published private(set) var samplesSaved = 0
    @Published private(set) var streamStartDate: Date?
    @Published private(set) var linkLost = false  // the Mac dropped out while streaming
    @Published private(set) var batteryLevel = -1.0  // 0...1, -1 when unknown
    @Published private(set) var lowPowerMode = false
    @Published private(set) var motionAvailable = true
    @Published private(set) var recordings: [URL] = []

    private let motionManager = CMMotionManager()
    private let motionQueue = OperationQueue()
    private var peripheralManager: CBPeripheralManager!
    private var dataCharacteristic: CBMutableCharacteristic!
    private var controlCharacteristic: CBMutableCharacteristic!
    private var serviceReady = false
    private var wantsAdvertising = true
    private var pendingSamples: [MotionSample] = []
    private var sequence: UInt16 = 0
    private var streamStartTimestamp: TimeInterval?
    private var backup: BackupWriter?
    private var motionRunning = false
    private var demoTimer: Timer?
    private var previewGravity: [(Double, Double, Double)] = []
    private var previewSum = (0.0, 0.0, 0.0)
    private var previewCount = 0
    private var rateWindowStart: TimeInterval?
    private var rateWindowCount = 0
    private var savedBrightness: CGFloat?
    // Counted per sample, published with the preview (10 times a second) so the screen does not
    // redraw at the motion rate.
    private var savedCount = 0
    private var sentCount = 0
    private var batchCount = 0
    private var lastBatchSize = 0

    override init() {
        super.init()
        motionQueue.name = "RespiPhoneIMU.motion"
        motionQueue.maxConcurrentOperationCount = 1
        peripheralManager = CBPeripheralManager(delegate: self, queue: nil)
        UIDevice.current.isBatteryMonitoringEnabled = true
        NotificationCenter.default.addObserver(self, selector: #selector(powerChanged), name: UIDevice.batteryLevelDidChangeNotification, object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(powerChanged), name: .NSProcessInfoPowerStateDidChange, object: nil)
        powerChanged()
        updateIdleTimer()
        refreshRecordings()
        startMotion()
    }

    // MARK: - Advertising and streaming

    func startAdvertising() {
        wantsAdvertising = true
        guard peripheralManager.state == .poweredOn else {
            statusMessage = "Bluetooth is not powered on."
            return
        }
        guard serviceReady else {
            configureService()
            statusMessage = "Preparing BLE service."
            return
        }
        peripheralManager.startAdvertising([
            CBAdvertisementDataLocalNameKey: "RespiPhoneIMU",
            CBAdvertisementDataServiceUUIDsKey: [imuServiceUUID],
        ])
        isAdvertising = true
        statusMessage = "Advertising RespiPhoneIMU."
    }

    func stopAdvertising() {
        wantsAdvertising = false
        peripheralManager.stopAdvertising()
        isAdvertising = false
        statusMessage = isStreaming ? "Streaming to subscribed central." : "Advertising stopped."
    }

    func startStreaming() {
        guard motionAvailable else {
            statusMessage = "Device motion is not available on this phone."
            return
        }
        guard !isStreaming else {
            return
        }
        pendingSamples.removeAll(keepingCapacity: true)
        sequence = 0
        streamStartTimestamp = nil
        savedCount = 0
        sentCount = 0
        batchCount = 0
        lastBatchSize = 0
        publishCounters()
        linkLost = false
        streamStartDate = Date()
        isStreaming = true
        startMotion()
        setDimmed(true)
        statusMessage = "Streaming motion samples."
    }

    func stopStreaming() {
        guard isStreaming else {
            return
        }
        isStreaming = false
        streamStartTimestamp = nil
        pendingSamples.removeAll(keepingCapacity: true)
        backup?.close()
        backup = nil
        streamStartDate = nil
        linkLost = false
        setDimmed(false)
        publishCounters()
        refreshRecordings()
        statusMessage = "Streaming stopped. The backup is under Saved on this phone."
    }

    func resetStatistics() {
        sentCount = 0
        batchCount = 0
        lastBatchSize = 0
        publishCounters()
        statusMessage = isStreaming ? "Streaming motion samples." : "Statistics reset."
    }

    /// Keep the preview running only while the app is on screen, unless a trial is being streamed.
    func sceneChanged(active: Bool) {
        if active {
            startMotion()
            refreshRecordings()
            updateIdleTimer()
        } else if isStreaming {
            statusMessage = "Keep the app open while streaming: iOS may pause motion updates in the background."
        } else {
            stopMotion()
        }
    }

    // MARK: - Saved recordings

    var recordingsFolder: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    func refreshRecordings() {
        let files = (try? FileManager.default.contentsOfDirectory(
            at: recordingsFolder,
            includingPropertiesForKeys: [.creationDateKey],
            options: [.skipsHiddenFiles]
        )) ?? []
        recordings = files
            .filter { $0.pathExtension == "csv" }
            .sorted { $0.lastPathComponent > $1.lastPathComponent }
    }

    func delete(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
        refreshRecordings()
    }

    // MARK: - Motion

    private func startMotion() {
        guard !motionRunning else {
            return
        }
        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 1.0 / max(1.0, sampleRateHz)
            motionManager.startDeviceMotionUpdates(to: motionQueue) { [weak self] motion, error in
                self?.handleMotion(motion, error: error)
            }
            motionRunning = true
            return
        }
        #if targetEnvironment(simulator)
        startDemoMotion()
        #else
        motionAvailable = false
        statusMessage = "Device motion is not available on this phone."
        #endif
    }

    private func stopMotion() {
        motionManager.stopDeviceMotionUpdates()
        demoTimer?.invalidate()
        demoTimer = nil
        motionRunning = false
    }

    private func restartMotion() {
        guard motionRunning, !isStreaming else {
            return
        }
        stopMotion()
        startMotion()
    }

    private func handleMotion(_ motion: CMDeviceMotion?, error: Error?) {
        if let error {
            DispatchQueue.main.async {
                self.statusMessage = error.localizedDescription
            }
            return
        }
        guard let motion else {
            return
        }
        let gravity = motion.gravity
        let user = motion.userAcceleration
        let rotation = motion.rotationRate
        let reading = MotionReading(
            timestamp: motion.timestamp,
            gravity: (gravity.x, gravity.y, gravity.z),
            acceleration: (gravity.x + user.x, gravity.y + user.y, gravity.z + user.z),
            rotationDegPerS: (rotation.x * 180.0 / .pi, rotation.y * 180.0 / .pi, rotation.z * 180.0 / .pi)
        )
        DispatchQueue.main.async {
            self.process(reading)
        }
    }

    #if targetEnvironment(simulator)
    /// The simulator has no motion sensors; a slow breath with a little noise lets the screens be checked.
    private func startDemoMotion() {
        let interval = 1.0 / max(1.0, sampleRateHz)
        demoTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            let now = ProcessInfo.processInfo.systemUptime
            let tilt = 0.004 * sin(2.0 * .pi * now / 4.5) + 0.0003 * Double.random(in: -1...1)
            let reading = MotionReading(
                timestamp: now,
                gravity: (tilt, 0.02, -0.999),
                acceleration: (tilt, 0.02, -0.999),
                rotationDegPerS: (0.2 * cos(2.0 * .pi * now / 4.5), 0.0, 0.0)
            )
            self?.process(reading)
        }
        motionRunning = true
    }
    #endif

    /// Runs on the main queue for every reading.
    private func process(_ reading: MotionReading) {
        updatePreview(reading)
        countRate(reading.timestamp)
        guard isStreaming else {
            return
        }
        if streamStartTimestamp == nil {
            streamStartTimestamp = reading.timestamp
            let secondsAgo = ProcessInfo.processInfo.systemUptime - reading.timestamp
            let startUnixMs = (Date().timeIntervalSince1970 - secondsAgo) * 1000.0
            backup = BackupWriter(directory: recordingsFolder, startUnixMs: startUnixMs)
            if backup == nil {
                statusMessage = "Could not create the backup file; streaming over Bluetooth only."
            }
        }
        let elapsedMs = max(0.0, (reading.timestamp - (streamStartTimestamp ?? reading.timestamp)) * 1000.0)
        let sample = MotionSample(
            timeMs: UInt32(clamping: Int(elapsedMs.rounded())),
            elapsedMs: elapsedMs,
            ax: reading.acceleration.x,
            ay: reading.acceleration.y,
            az: reading.acceleration.z,
            gx: reading.rotationDegPerS.x,
            gy: reading.rotationDegPerS.y,
            gz: reading.rotationDegPerS.z
        )
        backup?.append(sample)
        savedCount += 1
        guard subscriberCount > 0 else {
            return
        }
        pendingSamples.append(sample)
        flushPendingSamples()
    }

    /// Breathing tilts the chest, which shows as a slow change of the gravity direction.  The preview
    /// shows the gravity axis that moves most over the last 30 s, in mg.
    private func updatePreview(_ reading: MotionReading) {
        previewSum.0 += reading.gravity.x
        previewSum.1 += reading.gravity.y
        previewSum.2 += reading.gravity.z
        previewCount += 1
        guard Double(previewCount) >= max(1.0, sampleRateHz / previewRateHz) else {
            return
        }
        let n = Double(previewCount)
        publishCounters()
        previewGravity.append((previewSum.0 / n, previewSum.1 / n, previewSum.2 / n))
        previewSum = (0.0, 0.0, 0.0)
        previewCount = 0
        let keep = Int(previewRateHz * previewSeconds)
        if previewGravity.count > keep {
            previewGravity.removeFirst(previewGravity.count - keep)
        }
        let axes = [previewGravity.map(\.0), previewGravity.map(\.1), previewGravity.map(\.2)]
        let chosen = axes.max { variance($0) < variance($1) } ?? []
        let mean = chosen.reduce(0.0, +) / Double(max(chosen.count, 1))
        trace = chosen.map { ($0 - mean) * 1000.0 }
        let sorted = trace.sorted()
        if sorted.count >= 20 {
            traceSpanMg = sorted[Int(0.95 * Double(sorted.count - 1))] - sorted[Int(0.05 * Double(sorted.count - 1))]
        }
    }

    private func publishCounters() {
        samplesSaved = savedCount
        samplesSent = sentCount
        batchesSent = batchCount
        latestBatchSize = lastBatchSize
    }

    private func countRate(_ timestamp: TimeInterval) {
        guard let start = rateWindowStart else {
            rateWindowStart = timestamp
            rateWindowCount = 0
            return
        }
        rateWindowCount += 1
        if timestamp - start >= 1.0 {
            measuredRateHz = Double(rateWindowCount) / (timestamp - start)
            rateWindowStart = timestamp
            rateWindowCount = 0
        }
    }

    private func variance(_ values: [Double]) -> Double {
        guard values.count > 1 else {
            return 0.0
        }
        let mean = values.reduce(0.0, +) / Double(values.count)
        return values.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / Double(values.count)
    }

    // MARK: - Screen, power and alerts

    private func updateIdleTimer() {
        DispatchQueue.main.async {
            UIApplication.shared.isIdleTimerDisabled = self.keepScreenAwake
        }
    }

    private var screen: UIScreen? {
        UIApplication.shared.connectedScenes.compactMap { ($0 as? UIWindowScene)?.screen }.first
    }

    private func setDimmed(_ dimmed: Bool) {
        guard let screen else {
            return
        }
        if dimmed && dimScreenWhileStreaming {
            if savedBrightness == nil {
                savedBrightness = screen.brightness
            }
            screen.brightness = 0.05
        } else if let saved = savedBrightness {
            screen.brightness = saved
            savedBrightness = nil
        }
    }

    @objc private func powerChanged() {
        DispatchQueue.main.async {
            self.batteryLevel = Double(UIDevice.current.batteryLevel)
            self.lowPowerMode = ProcessInfo.processInfo.isLowPowerModeEnabled
        }
    }

    private func alertLinkLost() {
        UINotificationFeedbackGenerator().notificationOccurred(.error)
        AudioServicesPlaySystemSound(kSystemSoundID_Vibrate)
    }

    // MARK: - BLE

    private func configureService() {
        peripheralManager.removeAllServices()
        dataCharacteristic = CBMutableCharacteristic(
            type: imuDataUUID,
            properties: [.notify],
            value: nil,
            permissions: []
        )
        controlCharacteristic = CBMutableCharacteristic(
            type: imuControlUUID,
            properties: [.write, .writeWithoutResponse],
            value: nil,
            permissions: [.writeable]
        )
        let service = CBMutableService(type: imuServiceUUID, primary: true)
        service.characteristics = [dataCharacteristic, controlCharacteristic]
        serviceReady = false
        peripheralManager.add(service)
    }

    @discardableResult
    private func flushPendingSamples(force: Bool = false) -> Bool {
        guard dataCharacteristic != nil else {
            return false
        }
        guard subscriberCount > 0, !pendingSamples.isEmpty else {
            return false
        }

        let maximumCount = maxSamplesPerNotification()
        // Batch up to the central's negotiated payload size instead of one notification per
        // motion callback; each sample keeps its own timestamp.
        if !force && pendingSamples.count < maximumCount {
            return false
        }
        let count = min(maximumCount, pendingSamples.count)
        let batch = Array(pendingSamples.prefix(count))
        let data = encodeBatch(batch)
        let sent = peripheralManager.updateValue(data, for: dataCharacteristic, onSubscribedCentrals: nil)
        if sent {
            pendingSamples.removeFirst(count)
            sentCount += count
            batchCount += 1
            lastBatchSize = count
        }
        return sent
    }

    private func maxSamplesPerNotification() -> Int {
        let centralLimits = dataCharacteristic.subscribedCentrals?.map(\.maximumUpdateValueLength) ?? []
        let maximumBytes = centralLimits.min() ?? 20
        let payloadBytes = max(16, maximumBytes - 4)
        return max(1, min(12, payloadBytes / 16))
    }

    private func encodeBatch(_ samples: [MotionSample]) -> Data {
        var data = Data(capacity: 4 + samples.count * 16)
        data.appendUInt8(1)
        data.appendUInt8(UInt8(clamping: samples.count))
        data.appendLittleEndian(sequence)
        sequence &+= 1
        for sample in samples {
            data.appendLittleEndian(sample.timeMs)
            data.appendLittleEndian(clampInt16(sample.ax * 1000.0))
            data.appendLittleEndian(clampInt16(sample.ay * 1000.0))
            data.appendLittleEndian(clampInt16(sample.az * 1000.0))
            data.appendLittleEndian(clampInt16(sample.gx * 100.0))
            data.appendLittleEndian(clampInt16(sample.gy * 100.0))
            data.appendLittleEndian(clampInt16(sample.gz * 100.0))
        }
        return data
    }

    private func clampInt16(_ value: Double) -> Int16 {
        Int16(max(Double(Int16.min), min(Double(Int16.max), value.rounded())))
    }
}

extension MotionBLEStreamer: CBPeripheralManagerDelegate {
    func peripheralManagerDidUpdateState(_ peripheral: CBPeripheralManager) {
        switch peripheral.state {
        case .poweredOn:
            bluetoothState = "powered on"
            configureService()
        case .poweredOff:
            bluetoothState = "powered off"
            isAdvertising = false
            stopStreaming()
        case .unauthorized:
            bluetoothState = "unauthorized"
            statusMessage = "Bluetooth permission is not authorized."
        case .unsupported:
            bluetoothState = "unsupported"
            statusMessage = "BLE peripheral mode is not supported."
        case .resetting:
            bluetoothState = "resetting"
        case .unknown:
            bluetoothState = "unknown"
        @unknown default:
            bluetoothState = "unknown"
        }
    }

    func peripheralManager(_ peripheral: CBPeripheralManager, didAdd service: CBService, error: Error?) {
        if let error {
            statusMessage = error.localizedDescription
            serviceReady = false
            return
        }
        serviceReady = true
        if wantsAdvertising {
            startAdvertising()
        }
    }

    func peripheralManagerDidStartAdvertising(_ peripheral: CBPeripheralManager, error: Error?) {
        if let error {
            isAdvertising = false
            statusMessage = error.localizedDescription
        } else {
            isAdvertising = true
            statusMessage = "Advertising RespiPhoneIMU."
        }
    }

    func peripheralManager(_ peripheral: CBPeripheralManager, central: CBCentral, didSubscribeTo characteristic: CBCharacteristic) {
        subscriberCount = dataCharacteristic.subscribedCentrals?.count ?? 0
        statusMessage = "Central subscribed."
        if linkLost {
            linkLost = false
            UINotificationFeedbackGenerator().notificationOccurred(.success)
        }
        if autoStartOnConnection && !isStreaming {
            startStreaming()
        }
    }

    func peripheralManager(_ peripheral: CBPeripheralManager, central: CBCentral, didUnsubscribeFrom characteristic: CBCharacteristic) {
        subscriberCount = dataCharacteristic.subscribedCentrals?.count ?? 0
        statusMessage = subscriberCount > 0 ? "Central subscribed." : "No subscribed central."
        if isStreaming && subscriberCount == 0 {
            linkLost = true
            alertLinkLost()
        }
    }

    func peripheralManagerIsReady(toUpdateSubscribers peripheral: CBPeripheralManager) {
        while flushPendingSamples(force: true) {}
    }

    func peripheralManager(_ peripheral: CBPeripheralManager, didReceiveWrite requests: [CBATTRequest]) {
        for request in requests {
            guard request.characteristic.uuid == imuControlUUID else {
                peripheral.respond(to: request, withResult: .requestNotSupported)
                continue
            }
            let command = String(data: request.value ?? Data(), encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .uppercased()
            DispatchQueue.main.async {
                switch command {
                case "START":
                    self.startStreaming()
                case "STOP":
                    self.stopStreaming()
                default:
                    self.statusMessage = "Unknown control command."
                }
            }
            peripheral.respond(to: request, withResult: .success)
        }
    }
}

private extension Data {
    mutating func appendUInt8(_ value: UInt8) {
        append(contentsOf: [value])
    }

    mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
        var littleEndian = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndian) { rawBuffer in
            append(contentsOf: rawBuffer)
        }
    }
}
