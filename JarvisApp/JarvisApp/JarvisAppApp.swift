import SwiftUI

@main
struct JarvisAppApp: App {

    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appDelegate.backendManager)
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentMinSize)
        .commands {
            // Replace default "New Window" with nothing
            CommandGroup(replacing: .newItem) {}

            // ── Jarvis menu ─────────────────────────────────────────────
            CommandMenu("Jarvis") {
                Button("Restart Backend") {
                    appDelegate.backendManager.restart()
                }
                .keyboardShortcut("r", modifiers: [.command, .shift])

                Button("Show Logs") {
                    appDelegate.showLogsWindow()
                }
                .keyboardShortcut("l", modifiers: [.command, .shift])

                Divider()

                Button("Reset Mic Permission…") {
                    appDelegate.resetMicrophonePermission()
                }
            }
        }
    }
}
