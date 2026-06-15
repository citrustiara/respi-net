import CoreBluetooth
import CoreMotion
import Foundation

private let imuServiceUUID = CBUUID(string: "7B61B4E2-F5B4-4C90-8C7F-A7B2F1E8F4D0")
private let imuDataUUID = CBUUID(string: "7B61B4E3-F5B4-4C90-8C7F-A7B2F1E8F4D0")
private let imuControlUUID = CBUUID(string: "7B61B4E4-F5B4-4C90-8C7F-A7B2F1E8F4D0")

private struct MotionSample {
    let timeMs: UInt32
    let ax: Double
    let ay: Double
    let az: Double
    let gx: Double
    let gy: Double
    let gz: Double
}

final class MotionBLEStreamer: NSObject, ObservableObject {
    @Published var bluetoothState = "starting"
    @Published var isAdvertising = false
    @Published var isStreaming = false
    @Published var subscriberCount = 0
    @Published var sampleRateHz = 100.0
    @Published var samplesSent = 0
    @Published var batchesSent = 0
    @Published var latestBatchSize = 0
    @Published var statusMessage = "Waiting for Bluetooth."

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

    override init() {
        super.init()
        motionQueue.name = "RespiPhoneIMU.motion"
        motionQueue.maxConcurrentOperationCount = 1
        peripheralManager = CBPeripheralManager(delegate: self, queue: nil)
    }

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
        guard motionManager.isDeviceMotionAvailable else {
            statusMessage = "Device motion is not available on this phone."
            return
        }
        guard !isStreaming else {
            return
        }

        pendingSamples.removeAll(keepingCapacity: true)
        sequence = 0
        streamStartTimestamp = nil
        samplesSent = 0
        batchesSent = 0
        latestBatchSize = 0
        motionManager.deviceMotionUpdateInterval = 1.0 / max(1.0, sampleRateHz)
        motionManager.startDeviceMotionUpdates(to: motionQueue) { [weak self] motion, error in
            self?.handleMotion(motion, error: error)
        }
        isStreaming = true
        statusMessage = "Streaming motion samples."
    }

    func stopStreaming() {
        guard isStreaming else {
            return
        }
        motionManager.stopDeviceMotionUpdates()
        isStreaming = false
        streamStartTimestamp = nil
        pendingSamples.removeAll(keepingCapacity: true)
        statusMessage = "Streaming stopped."
    }

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
        if streamStartTimestamp == nil {
            streamStartTimestamp = motion.timestamp
        }
        let elapsedMs = max(0.0, (motion.timestamp - (streamStartTimestamp ?? motion.timestamp)) * 1000.0)
        let gravity = motion.gravity
        let userAcceleration = motion.userAcceleration
        let rotationRate = motion.rotationRate
        let sample = MotionSample(
            timeMs: UInt32(clamping: Int(elapsedMs.rounded())),
            ax: gravity.x + userAcceleration.x,
            ay: gravity.y + userAcceleration.y,
            az: gravity.z + userAcceleration.z,
            gx: rotationRate.x * 180.0 / .pi,
            gy: rotationRate.y * 180.0 / .pi,
            gz: rotationRate.z * 180.0 / .pi
        )
        DispatchQueue.main.async {
            guard self.subscriberCount > 0 else {
                return
            }
            self.pendingSamples.append(sample)
            self.flushPendingSamples()
        }
    }

    @discardableResult
    private func flushPendingSamples(force: Bool = false) -> Bool {
        guard dataCharacteristic != nil else {
            return false
        }
        guard subscriberCount > 0, !pendingSamples.isEmpty else {
            return false
        }

        let count = min(maxSamplesPerNotification(), pendingSamples.count)
        if !force && pendingSamples.count < count {
            return false
        }
        let batch = Array(pendingSamples.prefix(count))
        let data = encodeBatch(batch)
        let sent = peripheralManager.updateValue(data, for: dataCharacteristic, onSubscribedCentrals: nil)
        if sent {
            pendingSamples.removeFirst(count)
            DispatchQueue.main.async {
                self.samplesSent += count
                self.batchesSent += 1
                self.latestBatchSize = count
                self.statusMessage = "Streaming \(self.samplesSent) samples."
            }
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
    }

    func peripheralManager(_ peripheral: CBPeripheralManager, central: CBCentral, didUnsubscribeFrom characteristic: CBCharacteristic) {
        subscriberCount = dataCharacteristic.subscribedCentrals?.count ?? 0
        statusMessage = subscriberCount > 0 ? "Central subscribed." : "No subscribed central."
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
        withUnsafeBytes(of: &littleEndian) { rawBuffer in
            append(contentsOf: rawBuffer)
        }
    }
}
