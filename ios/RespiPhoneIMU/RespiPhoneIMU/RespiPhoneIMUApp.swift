import SwiftUI

@main
struct RespiPhoneIMUApp: App {
    @StateObject private var streamer = MotionBLEStreamer()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(streamer)
        }
    }
}
