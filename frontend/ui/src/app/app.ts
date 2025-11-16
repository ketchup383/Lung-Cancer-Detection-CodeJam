import { Component, signal } from '@angular/core';
import { CancerDetection } from './cancer-detection/cancer-detection';

@Component({
  selector: 'app-root',
  imports: [CancerDetection],

  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  protected readonly title = signal('ui');
}
