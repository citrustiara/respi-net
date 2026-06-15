import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var streamer: MotionBLEStreamer

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 18) {
                statusPanel
                controls
                metrics
                Spacer(minLength: 0)
            }
            .padding(20)
            .navigationTitle("Respi IMU")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var statusPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(streamer.isAdvertising ? "Advertising" : "Not advertising", systemImage: "antenna.radiowaves.left.and.right")
                .font(.headline)
            Text(streamer.statusMessage)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Text("Bluetooth: \(streamer.bluetoothState)")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

    private var controls: some View {
        VStack(spacing: 14) {
            Button {
                streamer.isStreaming ? streamer.stopStreaming() : streamer.startStreaming()
            } label: {
                Label(streamer.isStreaming ? "Stop streaming" : "Start streaming", systemImage: streamer.isStreaming ? "stop.fill" : "play.fill")
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

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Label("Sample rate", systemImage: "waveform.path.ecg")
                    Spacer()
                    Text("\(Int(streamer.sampleRateHz)) Hz")
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                Slider(value: $streamer.sampleRateHz, in: 20...100, step: 10)
                    .disabled(streamer.isStreaming)
            }
        }
    }

    private var metrics: some View {
        Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 12) {
            GridRow {
                MetricTile(title: "Subscribers", value: "\(streamer.subscriberCount)", symbol: "personalhotspot")
                MetricTile(title: "Batch", value: "\(streamer.latestBatchSize)", symbol: "square.stack.3d.up")
            }
            GridRow {
                MetricTile(title: "Samples", value: "\(streamer.samplesSent)", symbol: "number")
                MetricTile(title: "Batches", value: "\(streamer.batchesSent)", symbol: "point.3.connected.trianglepath.dotted")
            }
        }
    }
}

private struct MetricTile: View {
    let title: String
    let value: String
    let symbol: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: symbol)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.title2.monospacedDigit().weight(.semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

#Preview {
    ContentView()
        .environmentObject(MotionBLEStreamer())
}
