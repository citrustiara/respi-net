import SwiftUI

@main
struct RespiPhoneIMUApp: App {
    @StateObject private var streamer = MotionBLEStreamer()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(streamer)
        }
        .onChange(of: scenePhase) { phase in
            streamer.sceneChanged(active: phase == .active)
        }
    }
}
