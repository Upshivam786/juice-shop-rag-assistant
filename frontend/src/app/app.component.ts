import { Component, signal } from '@angular/core'
import { dom } from '@fortawesome/fontawesome-svg-core'
import { RouterLink, RouterOutlet } from '@angular/router'
import { WelcomeComponent } from './welcome/welcome.component'
import { ChallengeSolvedNotificationComponent } from './challenge-solved-notification/challenge-solved-notification.component'
import { CtfSystemWideNotificationComponent } from './ctf-system-wide-notification/ctf-system-wide-notification.component'
import { ServerStartedNotificationComponent } from './server-started-notification/server-started-notification.component'
import { NavbarComponent } from './navbar/navbar.component'
import { SidenavComponent } from './sidenav/sidenav.component'
import { MatSidenavContainer, MatSidenav } from '@angular/material/sidenav'
import { MatIconModule } from '@angular/material/icon'
import { MatButtonModule } from '@angular/material/button'

dom.watch()

@Component({
  selector: 'app-root',
  standalone: true,
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss'],
  imports: [
    MatSidenavContainer,
    MatSidenav,
    MatIconModule,
    SidenavComponent,
    NavbarComponent,
    ServerStartedNotificationComponent,
    ChallengeSolvedNotificationComponent,
    CtfSystemWideNotificationComponent,
    WelcomeComponent,
    RouterOutlet,
    MatButtonModule,
    RouterLink
  ]
})
export class AppComponent {
  isChatOpen = signal(false)
}
