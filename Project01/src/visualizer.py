"""AI Color Memory Game: Game interface and visual feedback."""
import cv2
import numpy as np
from config import COLOR_BGR, TOUCH_HOLD_SECONDS, MAX_ROUND, ROUND_LENGTHS, ROUND_SCORES

FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(img, content, x, y, size=0.55, color=(240, 240, 240), thick=1):
    cv2.putText(img, content, (x, y), FONT, size, color, thick, cv2.LINE_AA)


def draw_screen(frame, regions, fingertip, hands, game, hover, progress, now,
                touch_armed=False, touch_status=""):
    board = frame.copy()
    h, w = board.shape[:2]
    for name, item in regions.items():
        color = COLOR_BGR[name]
        cv2.drawContours(board, [item["contour"]], -1, color, 3)
        cx, cy = item["center"]
        cv2.circle(board, (cx, cy), 4, (255, 255, 255), -1)
        label(board, name, max(0, cx - 30), max(15, cy - 12), 0.55, color, 2)
    if hover and hover in regions:
        item = regions[hover]
        cv2.drawContours(board, [item["contour"]], -1, (255, 255, 255), 4)
        if progress:
            cx, cy = item["center"]
            cv2.ellipse(board, (cx, cy), (19, 19), 0, -90, -90 + int(progress * 360),
                        (255, 255, 255), 3)
    label(board, f"Panels: {len(regions)}/4    Hands: {hands}", 12, 26, 0.62, (255, 255, 255), 2)
    label(board, "SPACE: Start | R: Reset | S: Save | H: Masks | Q: Quit", 12, h - 15, 0.43, (255, 255, 255), 1)

    hud = np.full((h, 370, 3), (35, 29, 25), dtype=np.uint8)
    label(hud, "AI Color Memory Game", 22, 42, 0.78, (255, 255, 255), 2)
    label(hud, "202431880 Choi Eunsoo", 24, 66, 0.55, (170, 170, 170))
    cv2.line(hud, (20, 80), (350, 80), (85, 85, 85), 1)
    label(hud, f"ROUND  {game.round}/{MAX_ROUND}", 22, 112, 0.68)
    label(hud, f"SCORE  {game.score}", 195, 112, 0.68, (95, 230, 115), 2)
    label(hud, game.state.replace("_", " "), 22, 147, 0.83, (100, 222, 255), 2)
    if 1 <= game.round <= MAX_ROUND:
        label(hud,
              f"Goal: {ROUND_LENGTHS[game.round-1]} colors | Clear: +{ROUND_SCORES[game.round-1]}",
              22, 172, 0.46, (205, 205, 205))
    if game.state == "SHOW":
        color = game.flash_color(now)
        if color:
            cv2.circle(hud, (185, 237), 59, COLOR_BGR[color], -1)
            cv2.circle(hud, (185, 237), 62, (255, 255, 255), 2)
            size = cv2.getTextSize(color, FONT, 0.68, 2)[0][0]
            label(hud, color, 185 - size // 2, 245, 0.68, (20, 20, 20), 2)
        else:
            label(hud, "WATCH...", 117, 237, 0.7)
    elif game.state == "INPUT":
        label(hud, "YOUR TURN", 93, 201, 0.86, (100, 230, 160), 2)
        label(hud, f"Input: {game.index}/{len(game.sequence)}", 106, 237, 0.65)
        label(hud, f"Hover: {hover or 'NONE'}  Ready: {'YES' if touch_armed else 'NO'}",
              37, 306, 0.46, (240, 240, 240))
        label(hud, f"Hold {TOUCH_HOLD_SECONDS:.2f}s to select.", 30, 253, 0.43, (240, 240, 240))
        label(hud, "Move your hand completely outside all panels", 30, 270, 0.42, (240, 240, 240))
        label(hud, "before re-entering for accurate detection.", 30, 287, 0.42, (240, 240, 240))
        if game.last_choice is not None:
            last_text = (f"Last: {game.last_choice} +{game.last_award}"
                         if game.last_correct else f"Last: {game.last_choice} WRONG")
            info = last_text + (" | " + touch_status if touch_status else "")
            label(hud, info[:48], 30, 330, 0.39,
                  (100, 230, 160) if game.last_correct else (85, 85, 255), 1)
        else:
            label(hud, touch_status[:42], 30, 330, 0.40, (120, 220, 255))
    elif game.state == "READY":
        ready = len(regions) == 4
        label(hud, "LIVE COLOR TRACKING", 59, 202, 0.61, (100, 230, 160), 2)
        label(hud, f"Panels: {len(regions)}/4", 126, 235, 0.59,
              (100, 230, 160) if ready else (120, 220, 255), 2)
        label(hud, "Press SPACE to start" if ready else "Show all 4 colored papers",
              66 if ready else 43, 266, 0.53, (240, 240, 240))
        label(hud, "Move your hand completely outside all panels", 30, 306, 0.42, (220, 220, 220))
        label(hud, "before re-entering for accurate detection.", 30, 325, 0.42, (220, 220, 220))
    elif game.state == "GAME_CLEAR":
        label(hud, "GAME CLEAR!", 67, 225, 0.93, (100, 230, 160), 2)
        label(hud, f"FINAL SCORE: {game.score}/100", 82, 260, 0.62, (100, 230, 160), 2)
    elif game.state == "GAME_OVER":
        label(hud, "GAME OVER", 79, 225, 0.93, (85, 85, 255), 2)
        label(hud, f"SCORE: {game.score}/100", 102, 260, 0.62, (240, 240, 240), 2)
    else:
        label(hud, "ROUND CLEAR!", 70, 225, 0.85, (100, 230, 160), 2)
        if 1 <= game.round <= MAX_ROUND:
            label(hud, f"ROUND TOTAL: {ROUND_SCORES[game.round-1]}", 69, 260,
                  0.62, (100, 230, 160), 2)
    label(hud, "Selections:", 24, 354, 0.54)
    for i, name in enumerate(game.entered):
        x = 35 + i * 45
        cv2.circle(hud, (x, 381), 12, COLOR_BGR[name], -1)
        label(hud, str(i + 1), x - 4, 386, 0.38, (20, 20, 20), 1)
    cv2.line(hud, (20, 402), (350, 402), (85, 85, 85), 1)
    # Wrap status into short safe lines.
    status = game.message
    if len(status) > 38:
        p = status.rfind(" ", 0, 38)
        if p < 1: p = 38
        label(hud, status[:p], 22, 425, 0.40)
        label(hud, status[p:].strip()[:38], 22, 444, 0.40)
    else:
        label(hud, status, 22, 433, 0.40)
    label(hud, "GachonUniv On-Device AI", 22, h - 15, 0.43, (160, 160, 160))
    return np.hstack((board, hud))


def masks_preview(masks):
    names = ["RED", "BLUE", "GREEN", "YELLOW"]
    tiles = []
    for name in names:
        mask = masks[name]
        small = cv2.resize(mask, (320, 240))
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        label(small, name, 12, 24, 0.7, COLOR_BGR[name], 2)
        tiles.append(small)
    return np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))
