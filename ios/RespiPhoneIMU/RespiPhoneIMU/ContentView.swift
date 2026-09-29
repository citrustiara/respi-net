import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var streamer: MotionBLEStreamer

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    checklist
                    placementCheck
                    controls
                    savedRecordings
                }
                .padding(20)
            }
            .navigationTitle("Respi IMU")
            .navigationBarTitleDisplayMode(.inline)
        }
        .fullScreenCover(isPresented: Binding(get: { streamer.isStreaming }, set: { _ in })) {
            RecordingView()
                .environmentObject(streamer)
        }
    }

    // MARK: - Before a trial

    private var checklist: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Ready to record?")
                .font(.headline)
            ChecklistRow(
                state: streamer.bluetoothState == "powered on" ? .ok : .problem,
                text: "Bluetooth \(streamer.bluetoothState)"
            )
            ChecklistRow(
                state: streamer.isAdvertising ? .ok : .problem,
                text: streamer.isAdvertising ? "Visible to the Mac" : "Not visible: tap Advertise below"
            )
            ChecklistRow(
                state: streamer.subscriberCount > 0 ? .ok : .waiting,
                text: streamer.subscriberCount > 0 ? "Mac connected" : "Waiting for the Mac recorder to connect"
            )
            ChecklistRow(state: batteryState, text: batteryText)
            ChecklistRow(
                state: streamer.lowPowerMode ? .problem : .ok,
                text: streamer.lowPowerMode ? "Low Power Mode is on: turn it off, it can slow the stream" : "Low Power Mode off"
            )
            if !streamer.motionAvailable {
                ChecklistRow(state: .problem, text: "Motion sensors are not available")
            }
            Text(streamer.statusMessage)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

    private var batteryState: ChecklistRow.State {
        if streamer.batteryLevel < 0 {
            return .waiting
        }
        return streamer.batteryLevel >= 0.3 ? .ok : .problem
    }

    private var batteryText: String {
        guard streamer.batteryLevel >= 0 else {
            return "Battery level unknown"
        }
        let percent = Int((streamer.batteryLevel * 100).rounded())
        return percent >= 30 ? "Battery \(percent)%" : "Battery \(percent)%: charge before a long session"
    }

    private var placementCheck: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("Placement check", systemImage: "lungs")
                    .font(.headline)
                Spacer()
                Text(String(format: "%.1f mg", streamer.traceSpanMg))
                    .font(.subheadline.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            TraceView(values: streamer.trace, color: .accentColor)
                .frame(height: 110)
                .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            Text("Strap the phone firmly and breathe normally. The line should rise and fall smoothly with each breath; a flat or jumpy line means the strap is loose.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var controls: some View {
        VStack(spacing: 14) {
            Button {
                streamer.startStreaming()
            } label: {
                Label("Start streaming", systemImage: "play.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)

            Button {
                streamer.isAdvertising ? streamer.stopAdvertising() : streamer.startAdvertising()
            } label: {
                Label(streamer.isAdvertising ? "Stop advertising" : "Advertise BLE service", systemImage: "dot.radiowaves.left.and.right")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)

            VStack(alignment: .leading, spacing: 10) {
                Label("Sample rate", systemImage: "waveform.path.ecg")
                Picker("Sample rate", selection: $streamer.sampleRateHz) {
                    Text("25 Hz").tag(25.0)
                    Text("50 Hz").tag(50.0)
                    Text("100 Hz").tag(100.0)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }

            DisclosureGroup("Options") {
                VStack(spacing: 12) {
                    Toggle(isOn: $streamer.autoStartOnConnection) {
                        Label("Auto-start on connection", systemImage: "bolt.fill")
                    }
                    Toggle(isOn: $streamer.keepScreenAwake) {
                        Label("Keep screen awake", systemImage: "sun.max.fill")
                    }
                    Toggle(isOn: $streamer.dimScreenWhileStreaming) {
                        Label("Dim screen while streaming", systemImage: "moon.fill")
                    }
                    Button(role: .destructive) {
                        streamer.resetStatistics()
                    } label: {
                        Label("Reset statistics", systemImage: "arrow.counterclockwise")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                }
                .padding(.top, 12)
            }
        }
    }

    private var savedRecordings: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Saved on this phone", systemImage: "externaldrive")
                .font(.headline)
            Text("Every trial is also saved here, in case Bluetooth drops. Share a file, or find it in the Files app under On My iPhone › Respi IMU.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if streamer.recordings.isEmpty {
                Text("No trials yet.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            ForEach(streamer.recordings, id: \.self) { url in
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(url.deletingPathExtension().lastPathComponent)
                            .font(.subheadline.monospacedDigit())
                        Text(fileSize(url))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    ShareLink(item: url) {
                        Image(systemName: "square.and.arrow.up")
                    }
                    Menu {
                        Button(role: .destructive) {
                            streamer.delete(url)
                        } label: {
                            Label("Delete", systemImage: "trash")
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                    .padding(.leading, 8)
                }
                .padding(12)
                .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            }
        }
    }

    private func fileSize(_ url: URL) -> String {
        let bytes = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }
}

// MARK: - During a trial

/// Full screen while streaming: readable at a glance from a few metres, and hard to stop by accident.
private struct RecordingView: View {
    @EnvironmentObject private var streamer: MotionBLEStreamer
    @State private var holding = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 22) {
                HStack(spacing: 10) {
                    Circle()
                        .fill(.red)
                        .frame(width: 14, height: 14)
                    Text("STREAMING")
                        .font(.title3.weight(.bold))
                        .tracking(2)
                }
                TimelineView(.periodic(from: .now, by: 1)) { context in
                    Text(elapsed(at: context.date))
                        .font(.system(size: 72, weight: .semibold, design: .rounded).monospacedDigit())
                }
                linkBanner
                TraceView(values: streamer.trace, color: .green)
                    .frame(height: 150)
                HStack(spacing: 12) {
                    stat("Rate", String(format: "%.0f Hz", streamer.measuredRateHz))
                    stat("Saved", "\(streamer.samplesSaved)")
                    stat("Sent", "\(streamer.samplesSent)")
                }
                Spacer(minLength: 0)
                holdToStop
            }
            .foregroundStyle(.white)
            .padding(24)
        }
        .statusBarHidden()
    }

    private var linkBanner: some View {
        let (text, symbol, color): (String, String, Color) = {
            if streamer.linkLost {
                return ("Mac disconnected — still saving on this phone", "exclamationmark.triangle.fill", .red)
            }
            if streamer.subscriberCount == 0 {
                return ("No Mac connected — saving on this phone only", "iphone", .orange)
            }
            return ("Mac connected", "checkmark.circle.fill", .green)
        }()
        return Label(text, systemImage: symbol)
            .font(.headline)
            .multilineTextAlignment(.center)
            .padding(.vertical, 10)
            .padding(.horizontal, 14)
            .frame(maxWidth: .infinity)
            .background(color.opacity(0.25), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).stroke(color, lineWidth: 1))
    }

    private var holdToStop: some View {
        ZStack(alignment: .leading) {
            Capsule()
                .fill(Color.white.opacity(0.12))
            GeometryReader { geometry in
                Capsule()
                    .fill(Color.red.opacity(0.7))
                    .frame(width: holding ? geometry.size.width : 0)
                    .animation(holding ? .linear(duration: 2) : .easeOut(duration: 0.2), value: holding)
            }
            Text(holding ? "Keep holding…" : "Hold 2 s to stop")
                .font(.headline)
                .frame(maxWidth: .infinity)
        }
        .frame(height: 56)
        .contentShape(Capsule())
        .onLongPressGesture(minimumDuration: 2, maximumDistance: 60) {
            holding = false
            streamer.stopStreaming()
        } onPressingChanged: { pressing in
            holding = pressing
        }
    }

    private func stat(_ title: String, _ value: String) -> some View {
        VStack(spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.white.opacity(0.6))
            Text(value)
                .font(.title3.monospacedDigit().weight(.semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.6)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 10)
        .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

    private func elapsed(at date: Date) -> String {
        let seconds = max(0, Int(date.timeIntervalSince(streamer.streamStartDate ?? date)))
        return String(format: "%02d:%02d", seconds / 60, seconds % 60)
    }
}

// MARK: - Pieces

private struct ChecklistRow: View {
    enum State {
        case ok, waiting, problem
    }

    let state: State
    let text: String

    var body: some View {
        Label {
            Text(text)
                .fixedSize(horizontal: false, vertical: true)
        } icon: {
            switch state {
            case .ok:
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
            case .waiting:
                Image(systemName: "circle.dashed").foregroundStyle(.secondary)
            case .problem:
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
            }
        }
        .font(.subheadline)
    }
}

/// A line of the breathing preview.  The vertical scale never goes below 2 mg, so sensor noise on a
/// still phone stays flat instead of being stretched into something that looks like breathing.
private struct TraceView: View {
    let values: [Double]
    let color: Color

    var body: some View {
        GeometryReader { geometry in
            let low = values.min() ?? 0.0
            let high = values.max() ?? 0.0
            let span = max(high - low, 2.0)
            let middle = (high + low) / 2.0
            Path { path in
                for (index, value) in values.enumerated() {
                    let x = geometry.size.width * CGFloat(index) / CGFloat(max(values.count - 1, 1))
                    let y = geometry.size.height * (0.5 - CGFloat((value - middle) / span) * 0.9)
                    if index == 0 {
                        path.move(to: CGPoint(x: x, y: y))
                    } else {
                        path.addLine(to: CGPoint(x: x, y: y))
                    }
                }
            }
            .stroke(color, style: StrokeStyle(lineWidth: 2.5, lineCap: .round, lineJoin: .round))
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(MotionBLEStreamer())
}
