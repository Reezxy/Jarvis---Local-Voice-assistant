import SwiftUI

struct LogsView: View {
    @EnvironmentObject var backend: BackendManager
    @State private var autoScroll = true

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {

            // ── Toolbar ─────────────────────────────────────────────────
            HStack(spacing: 10) {
                Text("Backend Logs")
                    .font(.headline)
                Spacer()
                Toggle("Auto-scroll", isOn: $autoScroll)
                    .toggleStyle(.switch)
                    .controlSize(.small)
                Button("Clear") { backend.clearLogs() }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                Button("Restart") { backend.restart() }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .tint(.blue)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(Color(NSColor.windowBackgroundColor))

            Divider()

            // ── Scrolling log text ───────────────────────────────────────
            ScrollViewReader { proxy in
                ScrollView(.vertical) {
                    Text(backend.logs.isEmpty ? "(no output yet)" : backend.logs)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(backend.logs.isEmpty ? .secondary : .primary)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(8)
                        .id("bottom")
                }
                .onChange(of: backend.logs) { _ in
                    if autoScroll {
                        withAnimation(.none) {
                            proxy.scrollTo("bottom", anchor: .bottom)
                        }
                    }
                }
            }
        }
        .frame(minWidth: 560, minHeight: 320)
    }
}
