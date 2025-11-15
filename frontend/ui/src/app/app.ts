import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { CancerDetection } from './cancer-detection/cancer-detection';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, CancerDetection],

  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  protected readonly title = signal('ui');
}
